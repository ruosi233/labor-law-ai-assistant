from datetime import date
from decimal import Decimal

import pytest

from labor_law_ai.calculator import (
    CompensationType,
    MissingCalculationFields,
    calculate_compensation,
    calculate_n_months,
)


def test_n_months_rounding_rules() -> None:
    assert calculate_n_months(date(2020, 1, 1), date(2022, 6, 1))[0] == Decimal("2.5")
    assert calculate_n_months(date(2020, 1, 1), date(2022, 7, 1))[0] == Decimal("3")
    assert calculate_n_months(date(2024, 1, 1), date(2024, 3, 1))[0] == Decimal("0.5")


def test_calculate_n_compensation() -> None:
    result = calculate_compensation(
        start_date=date(2020, 1, 1),
        end_date=date(2022, 6, 1),
        average_monthly_salary="10000",
        compensation_type=CompensationType.N,
        local_average_salary="8000",
    )

    assert result.n_months == Decimal("2.5")
    assert result.n_compensation == Decimal("25000.00")
    assert result.total == Decimal("25000.00")


def test_n_plus_one_requires_last_month_salary() -> None:
    with pytest.raises(MissingCalculationFields) as error:
        calculate_compensation(
            start_date=date(2020, 1, 1),
            end_date=date(2022, 1, 1),
            average_monthly_salary="10000",
            compensation_type=CompensationType.N_PLUS_1,
        )

    assert "notice_monthly_salary" in error.value.fields[0]


def test_n_plus_one_uses_separate_notice_salary() -> None:
    result = calculate_compensation(
        start_date=date(2020, 1, 1),
        end_date=date(2022, 1, 1),
        average_monthly_salary="10000",
        compensation_type=CompensationType.N_PLUS_1,
        notice_monthly_salary="12000",
        local_average_salary="8000",
    )

    assert result.n_compensation == Decimal("20000.00")
    assert result.notice_pay == Decimal("12000.00")
    assert result.total == Decimal("32000.00")


def test_high_income_cap_and_two_n() -> None:
    result = calculate_compensation(
        start_date=date(2009, 1, 1),
        end_date=date(2025, 1, 1),
        average_monthly_salary="25000",
        compensation_type=CompensationType.TWO_N,
        local_average_salary="6000",
    )

    assert result.n_months == Decimal("12")
    assert result.salary_base == Decimal("18000.00")
    assert result.n_compensation == Decimal("216000.00")
    assert result.total == Decimal("432000.00")


def test_end_date_cannot_precede_start_date() -> None:
    with pytest.raises(ValueError, match="不能早于"):
        calculate_n_months(date(2025, 1, 1), date(2024, 1, 1))

