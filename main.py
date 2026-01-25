import sys
from analyzer import analyze_swing

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <user_video> <reference_video>")
        sys.exit(1)
    analyze_swing(sys.argv[1], sys.argv[2])
