#!/usr/bin/env python3
import sys
import math
import random
import csv
import threading
import queue
import time
import traceback

from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSlider, QFileDialog
)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
import serial
import serial.tools.list_ports
import numpy as np

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
FAKE_FS = 6000  # Hz, fake sampling rate for the demo
ADC_MAX = 4095
VREF = 3.3

# UART frame: 256 samples * 2 bytes = 512 bytes
SAMPLES_PER_FRAME = 256
FRAME_BYTES = SAMPLES_PER_FRAME * 2

# ------------------------------------------------------------
# Fake ADC source (~6 kS/s)
# ------------------------------------------------------------
class FakeSource:
    def __init__(self, buf_size=4000, vref=VREF, fs=FAKE_FS):
        self.dt = 1 / fs
        self.fs = fs
        self.t = 0.0
        self.buf = deque([0] * buf_size, maxlen=buf_size)
        self.noise = 15
        self.vref = vref
        self.adc_max = ADC_MAX
        self.freq = 80  # Hz sine

    def generate(self):
        """Generate new samples and update circular buffer."""
        for _ in range(60):
            v = 2048 + 700 * math.sin(2 * math.pi * self.freq * self.t)
            v += random.randint(-self.noise, self.noise)
            v = max(0, min(self.adc_max, int(v)))
            self.buf.append(v)
            self.t += self.dt
        return list(self.buf)

    def to_voltage(self, arr):
        """ADC counts → volts."""
        scale = self.vref / self.adc_max
        return [x * scale for x in arr]

# ------------------------------------------------------------
# Serial reader thread (background)
# ------------------------------------------------------------
class SerialReader(threading.Thread):
    """
    Continuously read from serial, yield complete frames (list of 12-bit samples).
    - Non-blocking to GUI: pushes frames to a Queue.
    - Automatically re-syncs if a partial read happens.
    """
    def __init__(self, port: str, baud: int, frame_bytes=FRAME_BYTES, q: queue.Queue = None):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.frame_bytes = frame_bytes
        self.q = q or queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self.ser = None
        self._reconnect_delay = 1.0  # seconds
        self._buf = bytearray()

    def stop(self):
        self._stop.set()

    def close_ser(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def open_ser(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            # flush input to try to start clean
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            return True
        except Exception as e:
            self.ser = None
            return False

    def run(self):
        while not self._stop.is_set():
            try:
                if not self.ser:
                    opened = self.open_ser()
                    if not opened:
                        time.sleep(self._reconnect_delay)
                        continue

                # Read a chunk
                chunk = self.ser.read(256)  # read up to 256 bytes at a time
                if not chunk:
                    # no data this iteration — loop back
                    continue
                self._buf.extend(chunk)

                # While we have at least one full frame, extract them
                while len(self._buf) >= self.frame_bytes:
                    frame_bytes = bytes(self._buf[:self.frame_bytes])
                    # Remove consumed bytes
                    del self._buf[:self.frame_bytes]

                    # Convert to samples (little-endian 16-bit, lower 12 bits valid)
                    samples = []
                    for i in range(0, len(frame_bytes), 2):
                        lo = frame_bytes[i]
                        hi = frame_bytes[i + 1]
                        raw = (hi << 8) | lo
                        adc12 = raw & 0x0FFF
                        samples.append(adc12)

                    # push latest frame: if queue full, drop oldest to not block GUI
                    try:
                        self.q.put_nowait(samples)
                    except queue.Full:
                        try:
                            _ = self.q.get_nowait()  # drop one
                        except Exception:
                            pass
                        try:
                            self.q.put_nowait(samples)
                        except Exception:
                            pass

            except Exception:
                # On any serial error, close and retry
                try:
                    traceback.print_exc()
                except Exception:
                    pass
                self.close_ser()
                time.sleep(self._reconnect_delay)

        # cleanup
        self.close_ser()

# ------------------------------------------------------------
# Trigger utilities
# ------------------------------------------------------------
def find_triggers(data, threshold, rising=True, max_found=3):
    """Return indices where threshold crossing happens."""
    out = []
    for i in range(len(data) - 1):
        if rising:
            if data[i] < threshold <= data[i + 1]:
                out.append(i)
        else:
            if data[i] > threshold >= data[i + 1]:
                out.append(i)
        if len(out) >= max_found:
            break
    return out

# ------------------------------------------------------------
# Main Window
# ------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Mini Scope (UART)")

        # ---- data / state ----
        self.src_fake = FakeSource()
        self.is_frozen = False
        self.fft_mode = False
        self.trigger_mode = "AUTO"     # AUTO / NORMAL
        self.trigger_rising = True
        self.threshold = 2048          # ADC counts
        self.cycles_to_show = 3

        self.last_data_volts = []      # last time-domain volts window (for cursors & saving)

        # Cursor state
        self.cursors_enabled = False
        self.cursor_vertical = True  # Vertical (time) vs Horizontal (voltage)

        # UART
        self.use_uart = False
        self.serial_reader = None
        self.serial_queue = queue.Queue(maxsize=8)

        # ===== UI =====
        root = QWidget()
        self.setCentralWidget(root)
        hbox = QHBoxLayout(root)

        # ---- Plot ----
        self.plot = pg.PlotWidget(background=None)
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot(pen=pg.mkPen('g', width=2))
        hbox.addWidget(self.plot, stretch=3)

        # ---- Cursors (hidden initially) ----
        self.cursor1 = pg.InfiniteLine(angle=90, movable=True, pen='y')
        self.cursor2 = pg.InfiniteLine(angle=90, movable=True, pen='c')
        self.plot.addItem(self.cursor1)
        self.plot.addItem(self.cursor2)
        self.cursor1.setVisible(False)
        self.cursor2.setVisible(False)
        self.cursor1.setValue(10)
        self.cursor2.setValue(100)
        self.cursor1.sigPositionChanged.connect(self.update_cursors)
        self.cursor2.sigPositionChanged.connect(self.update_cursors)

        # ---- Right panel ----
        controls = QVBoxLayout()
        hbox.addLayout(controls, stretch=1)

        # Freeze
        self.freeze_btn = QPushButton("Freeze")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.clicked.connect(self.toggle_freeze)
        controls.addWidget(self.freeze_btn)

        # FFT
        self.fft_btn = QPushButton("FFT")
        self.fft_btn.setCheckable(True)
        self.fft_btn.clicked.connect(self.toggle_fft)
        controls.addWidget(self.fft_btn)

        # Trigger Mode
        controls.addWidget(QLabel("Trigger Mode"))
        self.trig_select = QComboBox()
        self.trig_select.addItems(["AUTO", "NORMAL"])
        self.trig_select.currentIndexChanged.connect(self.update_trigger_mode)
        controls.addWidget(self.trig_select)

        # Trigger Level (ADC counts)
        controls.addWidget(QLabel("Trigger Level"))
        self.th_slider = QSlider(Qt.Orientation.Horizontal)
        self.th_slider.setRange(0, ADC_MAX)
        self.th_slider.setValue(self.threshold)
        self.th_slider.valueChanged.connect(self.update_threshold)
        controls.addWidget(self.th_slider)

        # COM Port
        controls.addWidget(QLabel("COM Port"))
        self.com_select = QComboBox()
        self.update_com_ports()
        controls.addWidget(self.com_select)

        # Baud
        controls.addWidget(QLabel("Baud Rate"))
        self.baud_select = QComboBox()
        self.baud_select.addItems(["9600", "57600", "115200", "230400", "460800", "921600"])
        self.baud_select.setCurrentText("115200")
        controls.addWidget(self.baud_select)

        # Use UART toggle
        self.use_uart_btn = QPushButton("Use UART Data")
        self.use_uart_btn.setCheckable(True)
        self.use_uart_btn.clicked.connect(self.toggle_uart)
        controls.addWidget(self.use_uart_btn)

        # Cycles
        controls.addWidget(QLabel("Cycles"))
        self.cycles_select = QComboBox()
        for i in range(1, 11):
            self.cycles_select.addItem(str(i))
        self.cycles_select.setCurrentText(str(self.cycles_to_show))
        self.cycles_select.currentIndexChanged.connect(self.update_cycles)
        controls.addWidget(self.cycles_select)

        # Save
        self.save_btn = QPushButton("Save CSV")
        self.save_btn.clicked.connect(self.save_csv)
        controls.addWidget(self.save_btn)

        # Cursor controls
        self.cursor_enable_btn = QPushButton("Enable Cursors")
        self.cursor_enable_btn.setCheckable(True)
        self.cursor_enable_btn.clicked.connect(self.toggle_cursors)
        controls.addWidget(self.cursor_enable_btn)

        controls.addWidget(QLabel("Cursor Orientation"))
        self.cursor_orient = QComboBox()
        self.cursor_orient.addItems(["Vertical", "Horizontal"])
        self.cursor_orient.currentIndexChanged.connect(self.update_cursor_orientation)
        controls.addWidget(self.cursor_orient)

        # Stretch to push the readout to the bottom
        controls.addStretch(1)

        # Cursor readout at the very bottom
        self.cursor_label = QLabel("")
        self.cursor_label.setStyleSheet("color: white;")
        controls.addWidget(self.cursor_label)

        # ---- Timer ----
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30)  # ~33 FPS

    # --------------------------------------------------------
    # UI handlers
    # --------------------------------------------------------
    def toggle_freeze(self):
        self.is_frozen = self.freeze_btn.isChecked()

    def toggle_fft(self):
        self.fft_mode = self.fft_btn.isChecked()
        # Hide cursors in FFT mode (indices/units differ)
        self.apply_cursor_visibility()

    def update_trigger_mode(self):
        self.trigger_mode = self.trig_select.currentText()

    def update_threshold(self):
        self.threshold = self.th_slider.value()

    def update_cycles(self):
        self.cycles_to_show = int(self.cycles_select.currentText())

    def update_com_ports(self):
        ports = serial.tools.list_ports.comports()
        self.com_select.clear()
        # Show device name and description if available
        for p in ports:
            label = f"{p.device} ({p.description})" if p.description else p.device
            # keep device string as the device only
            self.com_select.addItem(p.device)

    def toggle_cursors(self):
        self.cursors_enabled = self.cursor_enable_btn.isChecked()
        self.apply_cursor_visibility()
        self.update_cursors()

    def update_cursor_orientation(self):
        self.cursor_vertical = (self.cursor_orient.currentText() == "Vertical")
        # Vertical = time cursors
        if self.cursor_vertical:
            self.cursor1.setAngle(90)
            self.cursor2.setAngle(90)
            # Place roughly inside current x-range
            self.cursor1.setValue(0)
            self.cursor2.setValue(100)
        else:
            self.cursor1.setAngle(0)
            self.cursor2.setAngle(0)
            # Place to reasonable volt positions
            self.cursor1.setValue(1.0)
            self.cursor2.setValue(2.0)
        self.apply_cursor_visibility()
        self.update_cursors()

    def apply_cursor_visibility(self):
        visible = self.cursors_enabled and (not self.fft_mode)
        self.cursor1.setVisible(visible)
        self.cursor2.setVisible(visible)

    # --------------------------------------------------------
    # UART handling
    # --------------------------------------------------------
    def toggle_uart(self):
        self.use_uart = self.use_uart_btn.isChecked()
        if self.use_uart:
            port = self.com_select.currentText()
            baud = int(self.baud_select.currentText())
            # clear any old queue items
            with self.serial_queue.mutex:
                self.serial_queue.queue.clear()
            # create and start reader
            self.serial_reader = SerialReader(port=port, baud=baud, frame_bytes=FRAME_BYTES, q=self.serial_queue)
            self.serial_reader.start()
            # small delay to allow thread to try open
            time.sleep(0.05)
        else:
            # stop reader
            if self.serial_reader:
                try:
                    self.serial_reader.stop()
                    self.serial_reader.join(timeout=0.5)
                except Exception:
                    pass
                self.serial_reader = None

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    def save_csv(self):
        if not self.last_data_volts:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
        if not fname:
            return
        try:
            with open(fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "voltage"])
                for i, v in enumerate(self.last_data_volts):
                    w.writerow([i, v])
        except Exception:
            traceback.print_exc()

    # --------------------------------------------------------
    # Cursors
    # --------------------------------------------------------
    def update_cursors(self):
        """Update cursor readout text for vertical/horizontal modes."""
        if not self.cursors_enabled or not self.last_data_volts or self.fft_mode:
            return

        N = len(self.last_data_volts)
        Fs = FAKE_FS

        if self.cursor_vertical:
            # Vertical cursors → time measurement
            x1 = self.cursor1.value()
            x2 = self.cursor2.value()
            # interpret x as sample index if reasonable, else clamp
            i1 = int(max(0, min(N - 1, x1)))
            i2 = int(max(0, min(N - 1, x2)))
            v1 = self.last_data_volts[i1]
            v2 = self.last_data_volts[i2]
            dv = v2 - v1
            dt = abs(x2 - x1) / Fs
            self.cursor_label.setText(
                f"P1: {i1} → {v1:.3f} V\n"
                f"P2: {i2} → {v2:.3f} V\n"
                f"ΔV = {dv:.3f} V\n"
                f"Δt = {dt:.6f} s"
            )
        else:
            # Horizontal cursors → voltage measurement (absolute lines)
            y1 = float(self.cursor1.value())
            y2 = float(self.cursor2.value())
            dv = abs(y2 - y1)
            self.cursor_label.setText(
                f"Y1 = {y1:.3f} V\n"
                f"Y2 = {y2:.3f} V\n"
                f"ΔV = {dv:.3f} V"
            )

    # --------------------------------------------------------
    # Plot update loop
    # --------------------------------------------------------
    def update_plot(self):
        if self.is_frozen:
            return

        # Get raw counts either from UART queue or from fake generator
        raw = None
        if self.use_uart:
            # consume all frames and use the latest to avoid backlog/jitter
            latest = None
            try:
                while True:
                    latest = self.serial_queue.get_nowait()
            except queue.Empty:
                pass
            if latest is None:
                return  # no new frame yet
            raw = latest
        else:
            raw = self.src_fake.generate()  # list of ADC counts

        # If raw is shorter than needed, bail
        if not raw:
            return

        triggers = find_triggers(raw, self.threshold, self.trigger_rising, max_found=3)

        # Build time-domain window (counts) according to trigger mode
        if self.trigger_mode == "AUTO":
            if len(triggers) < 2:
                data_counts = raw[-2000:] if len(raw) >= 2000 else raw[:]
            else:
                p0, p1 = triggers[:2]
                period = max(1, p1 - p0)
                needed = period * self.cycles_to_show
                # clamp to available length
                if p0 + needed <= len(raw):
                    data_counts = raw[p0:p0 + needed]
                else:
                    # fall back to last N
                    data_counts = raw[-needed:] if len(raw) >= needed else raw[:]
        else:  # NORMAL
            if len(triggers) < 2:
                return  # wait until valid trigger
            p0, p1 = triggers[:2]
            period = max(1, p1 - p0)
            needed = period * self.cycles_to_show
            if p0 + needed <= len(raw):
                data_counts = raw[p0:p0 + needed]
            else:
                data_counts = raw[-needed:] if len(raw) >= needed else raw[:]

        # Convert to volts
        if self.use_uart:
            scale = VREF / ADC_MAX
            volts = [x * scale for x in data_counts]
        else:
            volts = self.src_fake.to_voltage(data_counts)

        self.last_data_volts = volts

        if not self.fft_mode:
            # TIME DOMAIN
            self.plot.setLabel("bottom", "Samples")
            self.plot.setLabel("left", "Volts")
            self.plot.enableAutoRange()
            self.curve.setData(volts)
            # Keep cursor visibility up to date and refresh readout
            self.apply_cursor_visibility()
            self.update_cursors()
            return

        # FFT MODE (magnitude spectrum)
        Fs = FAKE_FS
        n = len(volts)
        if n < 16:
            return

        yf = np.fft.rfft(volts)
        xf = np.fft.rfftfreq(n, 1 / Fs)
        mag = np.abs(yf)

        self.plot.setLabel("bottom", "Frequency (Hz)")
        self.plot.setLabel("left", "Magnitude")
        self.plot.enableAutoRange()
        self.curve.setData(xf, mag)
        # Hide cursors in FFT mode
        self.apply_cursor_visibility()

    def closeEvent(self, ev):
        # ensure serial thread stops
        if self.serial_reader:
            try:
                self.serial_reader.stop()
                self.serial_reader.join(timeout=0.5)
            except Exception:
                pass
        ev.accept()

# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 650)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
