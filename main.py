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
FAKE_FS = 2880  # Hz, fake sampling rate for the demo
ADC_MAX = 4095
VREF = 3.3
NUM_CHANNELS = 2

# UART frame: 128 samples per channel, interleaved, 2 bytes each = 512 bytes
SAMPLES_PER_CHANNEL = 128
SAMPLES_PER_FRAME = SAMPLES_PER_CHANNEL * NUM_CHANNELS  # 256 total samples
FRAME_BYTES = SAMPLES_PER_FRAME * 2  # 512 bytes
BUFFER_SIZE = 8000  # larger buffer for each channel

# ------------------------------------------------------------
# Fake ADC source (~6 kS/s) - Dual Channel
# ------------------------------------------------------------
class FakeSource:
    def __init__(self, buf_size=BUFFER_SIZE, vref=VREF, fs=FAKE_FS):
        self.dt = 1 / fs
        self.fs = fs
        self.t = 0.0
        self.buf_ch1 = deque([0] * buf_size, maxlen=buf_size)
        self.buf_ch2 = deque([0] * buf_size, maxlen=buf_size)
        self.noise = 15
        self.vref = vref
        self.adc_max = ADC_MAX
        self.freq1 = 80   # Hz sine for channel 1
        self.freq2 = 120  # Hz sine for channel 2

    def generate(self):
        """Generate new samples and update circular buffers for both channels."""
        for _ in range(60):
            # Channel 1: 80 Hz sine
            v1 = 2048 + 700 * math.sin(2 * math.pi * self.freq1 * self.t)
            v1 += random.randint(-self.noise, self.noise)
            # Add random spikes (simulating noise glitches)
            if random.random() < 0.01:  # 1% chance
                v1 += random.choice([-1000, 1000])
            v1 = max(0, min(self.adc_max, int(v1)))
            self.buf_ch1.append(v1)
            
            # Channel 2: 120 Hz sine with different amplitude
            v2 = 2048 + 500 * math.sin(2 * math.pi * self.freq2 * self.t)
            v2 += random.randint(-self.noise, self.noise)
            # Add random spikes
            if random.random() < 0.01:
                v2 += random.choice([-1000, 1000])
            v2 = max(0, min(self.adc_max, int(v2)))
            self.buf_ch2.append(v2)
            
            self.t += self.dt
        return {'ch1': list(self.buf_ch1), 'ch2': list(self.buf_ch2)}

    def to_voltage(self, data_dict):
        """ADC counts → volts for both channels."""
        scale = self.vref / self.adc_max
        return {
            'ch1': [x * scale for x in data_dict['ch1']],
            'ch2': [x * scale for x in data_dict['ch2']]
        }

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
                    # Data is interleaved: {C1S1, C2S1, C1S2, C2S2, ...}
                    samples_ch1 = []
                    samples_ch2 = []
                    for i in range(0, len(frame_bytes), 4):  # step by 4 bytes (2 samples)
                        # Channel 1 sample
                        lo1 = frame_bytes[i]
                        hi1 = frame_bytes[i + 1]
                        raw1 = (hi1 << 8) | lo1
                        adc12_ch1 = raw1 & 0x0FFF
                        samples_ch1.append(adc12_ch1)
                        
                        # Channel 2 sample
                        lo2 = frame_bytes[i + 2]
                        hi2 = frame_bytes[i + 3]
                        raw2 = (hi2 << 8) | lo2
                        adc12_ch2 = raw2 & 0x0FFF
                        samples_ch2.append(adc12_ch2)
                    
                    frame_data = {'ch1': samples_ch1, 'ch2': samples_ch2}

                    # push latest frame: if queue full, drop oldest to not block GUI
                    try:
                        self.q.put_nowait(frame_data)
                    except queue.Full:
                        try:
                            _ = self.q.get_nowait()  # drop one
                        except Exception:
                            pass
                        try:
                            self.q.put_nowait(frame_data)
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
def moving_average(data, window_size=5):
    """Simple moving average filter."""
    if len(data) < window_size:
        return data
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return list(ret[window_size - 1:] / window_size)

def find_triggers(data, threshold, rising=True, max_found=3, min_width=3):
    """
    Return indices where threshold crossing happens.
    min_width: Number of consecutive samples that must be valid after crossing.
    """
    out = []
    N = len(data)
    for i in range(N - min_width):
        if rising:
            # Crossing: prev < th <= curr
            if data[i] < threshold <= data[i + 1]:
                # Check if it stays above threshold for min_width samples
                valid = True
                for k in range(1, min_width + 1):
                    if i + k >= N or data[i + k] < threshold:
                        valid = False
                        break
                if valid:
                    out.append(i)
        else:
            # Crossing: prev > th >= curr
            if data[i] > threshold >= data[i + 1]:
                # Check if it stays below threshold for min_width samples
                valid = True
                for k in range(1, min_width + 1):
                    if i + k >= N or data[i + k] > threshold:
                        valid = False
                        break
                if valid:
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
        
        # Dual-channel buffers
        self.buffer_ch1 = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
        self.buffer_ch2 = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
        
        # Trigger state - Independent per channel
        self.trigger_mode = "AUTO"     # AUTO / NORMAL
        self.trigger_rising = True
        self.threshold_ch1 = 2048      # ADC counts
        self.threshold_ch2 = 2048      # ADC counts
        
        # Scaling
        self.volts_per_div = 1.0       # V/div
        self.time_per_div = 0.001      # s/div (1ms)
        self.ch1_offset = 0.0          # vertical offset in volts
        self.ch2_offset = 0.0
        
        # Smart Auto-Scaling State
        self.y_min = 0.0
        self.y_max = 3.3
        self.last_expansion_time = time.time()
        self.auto_scale_timeout = 10.0  # seconds to wait before contracting

        # Channel enables
        self.ch1_enabled = True
        self.ch2_enabled = True

        self.last_data_volts = {}      # dict with 'ch1' and 'ch2' keys

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
        self.plot.addLegend()
        self.curve_ch1 = self.plot.plot(pen=pg.mkPen('g', width=2), name='Channel 1')
        self.curve_ch2 = self.plot.plot(pen=pg.mkPen('y', width=2), name='Channel 2')
        
        # Trigger level indicators (arrows on plot)
        self.trigger_line_ch1 = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g', width=1, style=Qt.PenStyle.DashLine))
        self.trigger_line_ch2 = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.trigger_line_ch1)
        self.plot.addItem(self.trigger_line_ch2)
        
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

        # Trigger Level CH1
        controls.addWidget(QLabel("Trigger Level CH1"))
        self.th_slider_ch1 = QSlider(Qt.Orientation.Horizontal)
        self.th_slider_ch1.setRange(0, ADC_MAX)
        self.th_slider_ch1.setValue(self.threshold_ch1)
        self.th_slider_ch1.valueChanged.connect(self.update_threshold_ch1)
        controls.addWidget(self.th_slider_ch1)
        self.th_label_ch1 = QLabel(f"{self.threshold_ch1 * VREF / ADC_MAX:.3f} V")
        self.th_label_ch1.setStyleSheet("color: green;")
        controls.addWidget(self.th_label_ch1)
        
        # Trigger Level CH2
        controls.addWidget(QLabel("Trigger Level CH2"))
        self.th_slider_ch2 = QSlider(Qt.Orientation.Horizontal)
        self.th_slider_ch2.setRange(0, ADC_MAX)
        self.th_slider_ch2.setValue(self.threshold_ch2)
        self.th_slider_ch2.valueChanged.connect(self.update_threshold_ch2)
        controls.addWidget(self.th_slider_ch2)
        self.th_label_ch2 = QLabel(f"{self.threshold_ch2 * VREF / ADC_MAX:.3f} V")
        self.th_label_ch2.setStyleSheet("color: yellow;")
        controls.addWidget(self.th_label_ch2)

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

        # Time Scale Control (slider in ms)
        controls.addWidget(QLabel("Time Scale (ms/div)"))
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(1, 100)  # 0.1ms to 10ms (in 0.1ms units)
        self.time_slider.setValue(10)  # default 1ms
        self.time_slider.valueChanged.connect(self.update_time_scale)
        controls.addWidget(self.time_slider)
        self.time_label = QLabel("1.0 ms/div")
        self.time_label.setStyleSheet("color: white;")
        controls.addWidget(self.time_label)
        
        # Channel Enable
        self.ch1_enable_btn = QPushButton("CH1 Enabled")
        self.ch1_enable_btn.setCheckable(True)
        self.ch1_enable_btn.setChecked(True)
        self.ch1_enable_btn.clicked.connect(self.toggle_ch1)
        controls.addWidget(self.ch1_enable_btn)
        
        self.ch2_enable_btn = QPushButton("CH2 Enabled")
        self.ch2_enable_btn.setCheckable(True)
        self.ch2_enable_btn.setChecked(True)
        self.ch2_enable_btn.clicked.connect(self.toggle_ch2)
        controls.addWidget(self.ch2_enable_btn)

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

    def update_threshold_ch1(self):
        self.threshold_ch1 = self.th_slider_ch1.value()
        # Update voltage label
        voltage = self.threshold_ch1 * VREF / ADC_MAX
        self.th_label_ch1.setText(f"{voltage:.3f} V")
        # Update trigger line on plot
        self.trigger_line_ch1.setValue(voltage)

    def update_threshold_ch2(self):
        self.threshold_ch2 = self.th_slider_ch2.value()
        # Update voltage label
        voltage = self.threshold_ch2 * VREF / ADC_MAX
        self.th_label_ch2.setText(f"{voltage:.3f} V")
        # Update trigger line on plot
        self.trigger_line_ch2.setValue(voltage)
    
    def update_time_scale(self):
        # Slider value 1-100 represents 0.1ms to 10ms
        self.time_per_div = self.time_slider.value() / 1000.0  # convert to seconds
        self.time_label.setText(f"{self.time_slider.value() / 10.0:.1f} ms/div")
    
    def toggle_ch1(self):
        self.ch1_enabled = self.ch1_enable_btn.isChecked()
        self.curve_ch1.setVisible(self.ch1_enabled)
    
    def toggle_ch2(self):
        self.ch2_enabled = self.ch2_enable_btn.isChecked()
        self.curve_ch2.setVisible(self.ch2_enabled)


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
        if not self.last_data_volts or 'ch1' not in self.last_data_volts:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
        if not fname:
            return
        try:
            with open(fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "channel1_voltage", "channel2_voltage"])
                ch1 = self.last_data_volts.get('ch1', [])
                ch2 = self.last_data_volts.get('ch2', [])
                max_len = max(len(ch1), len(ch2))
                for i in range(max_len):
                    v1 = ch1[i] if i < len(ch1) else 0.0
                    v2 = ch2[i] if i < len(ch2) else 0.0
                    w.writerow([i, v1, v2])
        except Exception:
            traceback.print_exc()

    # --------------------------------------------------------
    # Cursors
    # --------------------------------------------------------
    def update_cursors(self):
        """Update cursor readout text for vertical/horizontal modes."""
        if not self.cursors_enabled or not self.last_data_volts or self.fft_mode:
            return
        if 'ch1' not in self.last_data_volts:
            return

        ch1_data = self.last_data_volts['ch1']
        N = len(ch1_data)
        Fs = FAKE_FS

        if self.cursor_vertical:
            # Vertical cursors → time measurement
            x1 = self.cursor1.value()
            x2 = self.cursor2.value()
            # interpret x as sample index if reasonable, else clamp
            i1 = int(max(0, min(N - 1, x1)))
            i2 = int(max(0, min(N - 1, x2)))
            v1_ch1 = ch1_data[i1]
            v2_ch1 = ch1_data[i2]
            dv_ch1 = v2_ch1 - v1_ch1
            dt = abs(x2 - x1) / Fs
            self.cursor_label.setText(
                f"P1: {i1} → CH1:{v1_ch1:.3f} V\n"
                f"P2: {i2} → CH1:{v2_ch1:.3f} V\n"
                f"ΔV(CH1) = {dv_ch1:.3f} V\n"
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

        # Get raw counts (dict with 'ch1' and 'ch2') from UART queue or fake generator
        raw_dict = None
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
            raw_dict = latest
        else:
            raw_dict = self.src_fake.generate()  # dict of ADC counts

        # If raw is missing or invalid, bail
        if not raw_dict or 'ch1' not in raw_dict or 'ch2' not in raw_dict:
            return

        # Update larger buffers with new data
        for val in raw_dict['ch1']:
            self.buffer_ch1.append(val)
        for val in raw_dict['ch2']:
            self.buffer_ch2.append(val)

        # Work with buffer data (lists of ADC counts)
        raw_ch1 = list(self.buffer_ch1)
        raw_ch2 = list(self.buffer_ch2)

        # Time-Qualified Triggering Strategy (Glitch Rejection)
        # We use raw data directly but require the signal to hold for min_width samples.
        
        # Calculate samples to display based on time/div
        # Assume 10 divisions horizontal (typical oscilloscope)
        samples_to_show = int(self.time_per_div * FAKE_FS * 10)
        samples_to_show = max(100, min(samples_to_show, len(raw_ch1)))  # clamp

        # Independent triggering for each channel
        # Trigger on CH1 first (using RAW data with glitch rejection)
        triggers_ch1 = find_triggers(raw_ch1, self.threshold_ch1, self.trigger_rising, max_found=1, min_width=3)
        
        # Build CH1 data window according to trigger mode
        if self.trigger_mode == "AUTO":
            if len(triggers_ch1) >= 1:
                p0_ch1 = triggers_ch1[0]
                if p0_ch1 + samples_to_show <= len(raw_ch1):
                    data_ch1 = raw_ch1[p0_ch1:p0_ch1 + samples_to_show]
                else:
                    data_ch1 = raw_ch1[-samples_to_show:]
            else:
                # No trigger found, show last N samples
                data_ch1 = raw_ch1[-samples_to_show:]
        else:  # NORMAL
            if len(triggers_ch1) >= 1:
                p0_ch1 = triggers_ch1[0]
                if p0_ch1 + samples_to_show <= len(raw_ch1):
                    data_ch1 = raw_ch1[p0_ch1:p0_ch1 + samples_to_show]
                else:
                    # Not enough data, use what we have
                    data_ch1 = raw_ch1[-samples_to_show:]
            else:
                # In NORMAL mode, if no trigger, don't update
                data_ch1 = []

        # Now trigger on CH2 independently (using RAW data with glitch rejection)
        triggers_ch2 = find_triggers(raw_ch2, self.threshold_ch2, self.trigger_rising, max_found=1, min_width=3)
        
        # Build CH2 data window according to trigger mode
        if self.trigger_mode == "AUTO":
            if len(triggers_ch2) >= 1:
                p0_ch2 = triggers_ch2[0]
                if p0_ch2 + samples_to_show <= len(raw_ch2):
                    data_ch2 = raw_ch2[p0_ch2:p0_ch2 + samples_to_show]
                else:
                    data_ch2 = raw_ch2[-samples_to_show:]
            else:
                # No trigger found, show last N samples
                data_ch2 = raw_ch2[-samples_to_show:]
        else:  # NORMAL
            if len(triggers_ch2) >= 1:
                p0_ch2 = triggers_ch2[0]
                if p0_ch2 + samples_to_show <= len(raw_ch2):
                    data_ch2 = raw_ch2[p0_ch2:p0_ch2 + samples_to_show]
                else:
                    # Not enough data, use what we have
                    data_ch2 = raw_ch2[-samples_to_show:]
            else:
                # In NORMAL mode, if no trigger, don't update
                data_ch2 = []

        # If in NORMAL mode and either channel has no trigger, bail
        if self.trigger_mode == "NORMAL" and (not data_ch1 or not data_ch2):
            return

        # Convert to volts
        if self.use_uart:
            scale = VREF / ADC_MAX
            volts_ch1 = [x * scale for x in data_ch1]
            volts_ch2 = [x * scale for x in data_ch2]
        else:
            temp_dict = {'ch1': data_ch1, 'ch2': data_ch2}
            volts_dict = self.src_fake.to_voltage(temp_dict)
            volts_ch1 = volts_dict['ch1']
            volts_ch2 = volts_dict['ch2']

        # Apply offsets
        volts_ch1 = [v + self.ch1_offset for v in volts_ch1]
        volts_ch2 = [v + self.ch2_offset for v in volts_ch2]

        self.last_data_volts = {'ch1': volts_ch1, 'ch2': volts_ch2}

        if not self.fft_mode:
            # TIME DOMAIN
            self.plot.setLabel("bottom", "Samples")
            self.plot.setLabel("left", "Volts")
            
            # --- Smart Auto-Scaling ---
            # 1. Find min/max of current data
            all_data = volts_ch1 + volts_ch2
            if all_data:
                curr_min = min(all_data)
                curr_max = max(all_data)
                
                # Add some margin
                margin = (curr_max - curr_min) * 0.1 if (curr_max - curr_min) > 0 else 0.1
                target_min = curr_min - margin
                target_max = curr_max + margin
                
                now = time.time()
                
                # Check for expansion
                expanded = False
                if target_min < self.y_min:
                    self.y_min = target_min
                    expanded = True
                if target_max > self.y_max:
                    self.y_max = target_max
                    expanded = True
                
                if expanded:
                    self.last_expansion_time = now
                else:
                    # Check for contraction (only if we haven't expanded recently)
                    if (now - self.last_expansion_time) > self.auto_scale_timeout:
                        # Contract slowly or jump? "Contract after delay" usually means we can now fit to the smaller signal
                        # We'll just set it to the target
                        self.y_min = target_min
                        self.y_max = target_max
                        self.last_expansion_time = now # Reset timer so we don't jitter if it's borderline

            # Disable auto-range and set manual range
            self.plot.disableAutoRange(axis=pg.ViewBox.YAxis)
            self.plot.setYRange(self.y_min, self.y_max, padding=0)
            
            # Update both channel curves
            if self.ch1_enabled:
                self.curve_ch1.setData(volts_ch1)
            if self.ch2_enabled:
                self.curve_ch2.setData(volts_ch2)
            
            # Keep cursor visibility up to date and refresh readout
            self.apply_cursor_visibility()
            self.update_cursors()
            return

        # FFT MODE (magnitude spectrum)
        Fs = FAKE_FS
        n = len(volts_ch1)
        if n < 16:
            return

        # FFT for both channels
        yf_ch1 = np.fft.rfft(volts_ch1)
        yf_ch2 = np.fft.rfft(volts_ch2)
        xf = np.fft.rfftfreq(n, 1 / Fs)
        mag_ch1 = np.abs(yf_ch1)
        mag_ch2 = np.abs(yf_ch2)

        self.plot.setLabel("bottom", "Frequency (Hz)")
        self.plot.setLabel("left", "Magnitude")
        self.plot.enableAutoRange()
        
        if self.ch1_enabled:
            self.curve_ch1.setData(xf, mag_ch1)
        if self.ch2_enabled:
            self.curve_ch2.setData(xf, mag_ch2)
        
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
