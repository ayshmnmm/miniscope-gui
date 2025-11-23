#!/usr/bin/env python3
import serial
import time
import struct

PORT = "/dev/ttyUSB0"   # change if needed
BAUD = 9600
FRAME_BYTES = 512       # 256 samples * 2

def main():
    print(f"Opening {PORT} @ {BAUD}…")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print("FAILED to open port:", e)
        return

    ser.reset_input_buffer()
    print("Port opened.")

    while True:
        # Try reading 512 bytes
        chunk = ser.read(FRAME_BYTES)

        if len(chunk) == 0:
            print("No data… (timeout)")
            continue

        if len(chunk) != FRAME_BYTES:
            print(f"Partial frame: {len(chunk)} bytes")
            continue

        # Decode the first few samples
        samples = []
        for i in range(0, 16, 2):  # first 8 samples
            lo = chunk[i]
            hi = chunk[i+1]
            val = (hi << 8) | lo
            samples.append(val & 0x0FFF)

        print(f"Got frame {len(chunk)} bytes | First samples: {samples[:8]}")
        time.sleep(0.1)

if __name__ == "__main__":
    main()
