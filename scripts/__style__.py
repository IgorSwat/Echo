# ANSI escape codes for coloring
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")

def print_summary_line(label, value, color=Colors.ENDC):
    print(f"{label}: {color}{value}{Colors.ENDC}")

def print_mismatch_example(index, word, target, predicted, distance):
    print(f"\n{index}. Word: {Colors.BOLD}{word}{Colors.ENDC}")
    print(f"   Target:    {Colors.OKGREEN}{target}{Colors.ENDC}")
    print(f"   Predicted: {Colors.FAIL}{predicted}{Colors.ENDC}")
    print(f"   Distance:  {distance}")


def print_test_title(text):
    print()
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print()


def print_section(text):
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}--- {text} ---{Colors.ENDC}")


def print_info(label, value, color=Colors.ENDC):
    print(f"  {Colors.BOLD}{label}:{Colors.ENDC} {color}{value}{Colors.ENDC}")


def print_success(text):
    print(f"{Colors.OKGREEN}{Colors.BOLD}[PASS]{Colors.ENDC} {text}")


def print_error(text):
    print(f"{Colors.FAIL}{Colors.BOLD}[FAIL]{Colors.ENDC} {text}")


def print_separator(char="─", width=60):
    print(char * width)


def print_result(passed, total):
    print()
    if passed == total:
        print(f"{Colors.OKGREEN}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}  ALL TESTS PASSED ({passed}/{total}){Colors.ENDC}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.FAIL}{Colors.BOLD}  FAILED: {passed}/{total} passed{Colors.ENDC}")
        print(f"{Colors.FAIL}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    return passed == total
