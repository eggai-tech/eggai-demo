import re


def get_amount_score(expected: str, actual: str) -> tuple[float, bool]:
    try:
        expected_amount_match = re.search(r"\$(\d+(?:\.\d+)?)", expected)
        expected_amount = (
            expected_amount_match.group(1) if expected_amount_match else None
        )

        if not expected_amount:
            return (0.0, False)

        actual_amount_match = re.search(r"\$(\d+(?:\.\d+)?)", actual)
        actual_amount = actual_amount_match.group(1) if actual_amount_match else None

        if not actual_amount:
            return (0.0, True)

        expected_float = float(expected_amount)
        actual_float = float(actual_amount)

        if expected_float == actual_float:
            if expected_amount == actual_amount:
                return (1.0, True)
            else:
                return (0.8, True)
        else:
            return (0.0, True)

    except (IndexError, ValueError):
        return (0.0, True)


def get_date_score(expected: str, actual: str) -> tuple[float, bool]:
    expected_dates = re.findall(r"(\d{4}-\d{2}-\d{2})", expected)

    if not expected_dates:
        return (0.0, False)

    for expected_date in expected_dates:
        if expected_date in actual:
            return (1.0, True)

        year, month, day = expected_date.split("-")

        date_patterns = [
            f"{month}/{day}/{year}",  # 05/15/2026
            f"{year}/{month}/{day}",  # 2026/05/15
            f"{month}-{day}-{year}",  # 05-15-2026
            f"{month}.{day}.{year}",  # 05.15.2026
        ]

        for pattern in date_patterns:
            if pattern in actual:
                return (0.7, True)

        month_names = {
            "01": ["january", "jan"],
            "02": ["february", "feb"],
            "03": ["march", "mar"],
            "04": ["april", "apr"],
            "05": ["may"],
            "06": ["june", "jun"],
            "07": ["july", "jul"],
            "08": ["august", "aug"],
            "09": ["september", "sep", "sept"],
            "10": ["october", "oct"],
            "11": ["november", "nov"],
            "12": ["december", "dec"],
        }

        if month in month_names:
            for month_name in month_names[month]:
                if month_name.lower() in actual.lower() and year in actual:
                    return (0.7, True)

    return (0.0, True)


def get_status_score(expected: str, actual: str) -> tuple[float, bool]:
    status_patterns = [
        r"status is ['\"]([^'\"]+)['\"]",  # status is 'Paid'
        r"status is ([A-Za-z]+)",  # status is Paid
        r"status:? ['\"]([^'\"]+)['\"]",  # status: 'Paid'
        r"status:? ([A-Za-z]+)",  # status: Paid
    ]

    expected_status = None
    expected_has_quotes = False
    for pattern in status_patterns:
        match = re.search(pattern, expected, re.IGNORECASE)
        if match:
            expected_status = match.group(1).strip().lower()
            if "'" in pattern or '"' in pattern:
                expected_has_quotes = "'" in match.group(0) or '"' in match.group(0)
            break

    if not expected_status:
        return (0.0, False)

    actual_status = None
    actual_has_quotes = False
    for pattern in status_patterns:
        match = re.search(pattern, actual, re.IGNORECASE)
        if match:
            actual_status = match.group(1).strip().lower()
            if "'" in pattern or '"' in pattern:
                actual_has_quotes = "'" in match.group(0) or '"' in match.group(0)
            break

    if not actual_status:
        return (0.0, True)

    if expected_status == actual_status:
        if expected_has_quotes and actual_has_quotes:
            expected_quote = "'" if "'" in expected else '"'
            actual_quote = "'" if "'" in actual else '"'
            if expected_quote != actual_quote:
                return (0.9, True)
            else:
                return (1.0, True)
        elif expected_has_quotes and not actual_has_quotes:
            return (0.8, True)
        else:
            return (1.0, True)
    else:
        return (0.0, True)


def get_billing_cycle_score(expected: str, actual: str) -> tuple[float, bool]:
    if "billing cycle" not in expected.lower():
        return (0.0, False)

    cycle_patterns = [
        r"billing cycle is ['\"]([^'\"]+)['\"]",  # billing cycle is 'Monthly'
        r"billing cycle is ([A-Za-z-]+)",  # billing cycle is Monthly
        r"billing cycle:? ['\"]([^'\"]+)['\"]",  # billing cycle: 'Monthly'
        r"billing cycle:? ([A-Za-z-]+)",  # billing cycle: Monthly
    ]

    expected_cycle = None
    for pattern in cycle_patterns:
        match = re.search(pattern, expected, re.IGNORECASE)
        if match:
            expected_cycle = match.group(1).strip().lower()
            break

    if not expected_cycle:
        return (0.0, True)

    actual_cycle = None
    for pattern in cycle_patterns:
        match = re.search(pattern, actual, re.IGNORECASE)
        if match:
            actual_cycle = match.group(1).strip().lower()
            break

    if actual_cycle and expected_cycle == actual_cycle:
        return (1.0, True)

    actual_lower = actual.lower()

    if "billing cycle" in actual_lower:
        return (0.8, True)

    if "cycle" in actual_lower and expected_cycle in actual_lower:
        return (0.7, True)

    if expected_cycle in actual_lower:
        return (0.3, True)

    return (0.0, True)


def get_format_score(expected: str, actual: str) -> tuple[float, bool]:
    format_score = 0.0
    format_checks = 0
    partial_match = False

    if "2026" in expected or re.search(r"20\d\d", expected):
        format_checks += 1
        if re.search(r"20\d\d-\d\d-\d\d", expected) and re.search(
            r"20\d\d-\d\d-\d\d", actual
        ):
            format_score += 1.0
        elif re.search(r"20\d\d", actual):
            format_score += 0.5
            partial_match = True

    if "$" in expected:
        format_checks += 1
        if re.search(r"\$\d+\.\d\d", expected) and re.search(r"\$\d+\.\d\d", actual):
            format_score += 1.0
        elif "$" in actual:
            format_score += 0.5
            partial_match = True

    if format_checks > 0:
        final_score = format_score / format_checks

        # Downgrade score when formats match but overall text structure differs significantly
        if final_score >= 1.0:
            expected_lower = expected.lower()
            actual_lower = actual.lower()

            if (
                abs(len(expected) - len(actual)) > len(expected) * 0.3
                or expected_lower[:20] != actual_lower[:20]
            ):
                partial_match = True

        if partial_match and final_score >= 1.0:
            final_score = 0.9
        return (final_score, True)

    return (0.0, False)


def precision_metric(expected_str: str, actual_str: str) -> float:
    expected = expected_str.lower()
    actual = actual_str.lower()

    score = 0.0
    check_count = 0

    evaluations = [
        get_amount_score(expected, actual),
        get_date_score(expected, actual),
        get_status_score(expected, actual),
        get_billing_cycle_score(expected, actual),
        get_format_score(expected, actual),
    ]

    for eval_score, eval_performed in evaluations:
        if eval_performed:
            score += eval_score
            check_count += 1

    if check_count == 0:
        return 0.0

    return score / check_count
