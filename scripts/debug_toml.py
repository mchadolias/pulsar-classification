# debug_toml.py
import sys

config_path = "model_config.toml"  # Change this if your file has a different name

try:
    with open(config_path, "r") as f:
        lines = f.readlines()

    print(f"Total lines in file: {len(lines)}")
    print("\nLines 45-55:")
    for i in range(44, min(55, len(lines))):
        line_num = i + 1
        line = lines[i].rstrip()
        print(f"{line_num:3d}: {repr(line)}")

    print("\nChecking for syntax errors...")

    # Check for common issues
    for i, line in enumerate(lines):
        line_num = i + 1
        line_stripped = line.strip()

        # Check for .number without leading zero
        if line_stripped and "." in line_stripped:
            parts = line_stripped.split(".")
            for j in range(len(parts) - 1):
                # Check if part before decimal is empty or ends with space
                if parts[j] == "" or parts[j].endswith(" "):
                    print(f"⚠️  Line {line_num}: Possible missing leading zero before decimal")

        # Check for arrays with missing commas
        if "[" in line_stripped and "]" in line_stripped:
            # Remove spaces and check for numbers without commas
            clean = line_stripped.replace(" ", "")
            if "[" in clean and "]" in clean:
                array_content = clean.split("[", 1)[1].split("]", 1)[0]
                if array_content and "." in array_content:
                    # Check for patterns like 1.02.03.0
                    numbers = array_content.split(".")
                    if len(numbers) > 1:
                        for k in range(1, len(numbers)):
                            if numbers[k] and numbers[k - 1] and not numbers[k - 1].endswith(","):
                                print(f"⚠️  Line {line_num}: Possible missing comma in array")

except FileNotFoundError:
    print(f"Error: File '{config_path}' not found")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
