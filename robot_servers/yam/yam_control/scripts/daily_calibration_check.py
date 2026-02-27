"""
Lightweight Daily Calibration Check Script

Reads joint angles from YAM robot arms and displays calibration status.
No Google Sheet sync, no credentials, no branch management.

Usage:
    # Read all 4 arms at once (default)
    python -m yam_control.scripts.daily_calibration_check

    # Read specific arm only
    python -m yam_control.scripts.daily_calibration_check --channel can_leader_l
    python -m yam_control.scripts.daily_calibration_check --channel can_follow_r

    # Read multiple specific arms
    python -m yam_control.scripts.daily_calibration_check --channel can_leader_l can_leader_r
"""

from datetime import datetime
import re
import socket
import subprocess
import sys
from typing import Literal

import tyro


def get_station_number() -> str:
    """
    Detect station number from hostname.

    Parses hostnames like:
    - gear-yam-desktop-09 -> "09"
    - gear-ax8-max-10 -> "10"

    Returns station number or prompts user if detection fails.
    """
    hostname = socket.gethostname()

    match = re.search(r"gear-(?:yam-desktop|ax8-max)-(\d+)", hostname)

    if match:
        station_num = match.group(1)
        print(f"  Detected Station: {station_num} (from hostname: {hostname})")
        return station_num
    else:
        print(f"  Could not detect station from hostname: {hostname}")
        station_num = input("Please enter station number (e.g., 09, 10): ").strip()
        return station_num


def read_arm_angles(
    channel: Literal["can_leader_l", "can_leader_r", "can_follow_l", "can_follow_r"],
) -> str:
    """
    Read joint angles from a single arm by calling print_joint_angles.py.

    Returns formatted multi-line string of joint angles, or error message.
    """
    print(f"  Reading {channel}...", end=" ")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "yam_control.scripts.print_joint_angles", "--channel", channel],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode != 0:
            print("FAILED")
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"

            error_lines = error_msg.split("\n")
            filtered_errors = [
                line
                for line in error_lines
                if not line.startswith("warning:") and "extra-build-dependencies" not in line
            ]
            clean_error = "\n".join(filtered_errors).strip() or "Reading failed"

            if "Permission denied" in clean_error:
                return "ERROR - Permission denied (check CAN interface access)"
            elif "in use" in clean_error.lower() or "busy" in clean_error.lower():
                return "ERROR - Arm in use by another process"
            elif "No such device" in clean_error or "Cannot find device" in clean_error:
                return "ERROR - CAN device not found (check connections)"
            elif "Connection" in clean_error or "connect" in clean_error.lower():
                return "ERROR - Failed to connect to arm"
            elif len(clean_error) > 100:
                return f"ERROR - {clean_error[:100]}..."
            elif clean_error:
                return f"ERROR - {clean_error}"
            else:
                return "ERROR - Reading failed"

        output = result.stdout

        if "Joint Angles:" in output:
            lines = output.split("\n")
            joint_section_started = False
            joint_lines = []

            for line in lines:
                if "Joint Angles:" in line:
                    joint_section_started = True
                    continue
                if joint_section_started and line.strip().startswith("Joint"):
                    clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line.strip())
                    joint_lines.append(clean_line)

            if joint_lines:
                print("OK")
                return "\n".join(joint_lines)

        print("Could not parse output")
        return "ERROR - Could not parse joint angles from output"

    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return "ERROR - Read timeout (>30s)"
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return f"ERROR - {str(e)[:100]}"


def check_calibration_warnings(arm_name: str, joint_angles_str: str) -> list[str]:
    """
    Check if any joints 1-6 are outside acceptable calibration threshold.

    Returns list of warning messages (empty if all OK).
    """
    warnings = []

    for line in joint_angles_str.strip().split("\n"):
        if line.startswith("Joint"):
            parts = line.split(":")
            if len(parts) == 2:
                joint_num_str = parts[0].replace("Joint", "").strip()
                angle_str = parts[1].strip().replace("°", "").strip()

                try:
                    joint_num = int(joint_num_str)
                    angle = float(angle_str)

                    if 1 <= joint_num <= 6 and abs(angle) > 1.0:
                        warnings.append(
                            f"  {arm_name} Joint {joint_num}: {angle:+.2f} deg (outside +/-1 deg threshold)"
                        )
                except (ValueError, IndexError):
                    continue

    return warnings


def main(
    channel: (
        list[Literal["can_leader_l", "can_leader_r", "can_follow_l", "can_follow_r"]] | None
    ) = None,
) -> None:
    """
    Lightweight daily calibration check - reads joint angles and displays results locally.

    No Google Sheet sync; just reads and reports.

    Args:
        channel: Specific arm(s) to read.
            Options: can_leader_l, can_leader_r, can_follow_l, can_follow_r.
            If None, reads all 4 arms.
    """
    print("=" * 70)
    print("Daily Calibration Check - YAM Robot Arms")
    print("=" * 70)

    print("\n[1/2] Detecting station...")
    station_number = get_station_number()

    print("\n[2/2] Reading arm calibration data...")

    if channel:
        arm_channels = channel
        if len(arm_channels) == 1:
            print(f"  Mode: Single arm ({arm_channels[0]})")
        else:
            print(f"  Mode: {len(arm_channels)} arms ({', '.join(arm_channels)})")
    else:
        print("  Mode: All 4 arms")
        arm_channels = ["can_leader_l", "can_leader_r", "can_follow_l", "can_follow_r"]

    arm_data: dict[str, str] = {}
    calibration_warnings: list[str] = []

    for ch in arm_channels:
        arm_data[ch] = read_arm_angles(ch)

        if not arm_data[ch].startswith("ERROR"):
            warnings = check_calibration_warnings(ch, arm_data[ch])
            calibration_warnings.extend(warnings)

    print("\n" + "=" * 70)
    print("Summary:")
    date_obj = datetime.now()
    print(f"  Date: {date_obj.month}/{date_obj.day}/{date_obj.year}")
    print(f"  Station: {station_number}")
    if channel:
        if len(channel) == 1:
            print("  Mode: Single arm")
        else:
            print(f"  Mode: {len(channel)} arms")
    else:
        print("  Mode: All arms")

    print("\n  Arm readings:")
    for ch, data in arm_data.items():
        status = "OK" if not data.startswith("ERROR") else "FAILED"
        print(f"    {ch}: {status}")
        if not data.startswith("ERROR"):
            for line in data.strip().split("\n"):
                print(f"      {line}")

    if calibration_warnings:
        print("\n" + "=" * 70)
        print("CALIBRATION WARNINGS:")
        print("=" * 70)
        print("The following joints are outside the +/-1 deg threshold:")
        print()
        for warning in calibration_warnings:
            print(warning)
        print()
        print("These arms may benefit from recalibration.")
        print("=" * 70)
    else:
        all_ok = all(not d.startswith("ERROR") for d in arm_data.values())
        if all_ok:
            print("\n  All joints within +/-1 deg threshold.")

    print("\nCalibration check complete!")


def cli():
    tyro.cli(main)


if __name__ == "__main__":
    cli()
