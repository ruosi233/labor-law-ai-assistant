"""N、N+1、2N 经济补偿/赔偿金额的确定性计算。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

from dateutil.relativedelta import relativedelta


MONEY = Decimal("0.01")


class CompensationType(str, Enum):
    N = "N"
    N_PLUS_1 = "N+1"
    TWO_N = "2N"


class MissingCalculationFields(ValueError):
    def __init__(self, fields: list[str]) -> None:
        self.fields = fields
        super().__init__(f"缺少计算字段：{', '.join(fields)}")


@dataclass(frozen=True)
class CompensationResult:
    compensation_type: str
    service_period: str
    n_months: Decimal
    original_average_monthly_salary: Decimal
    salary_base: Decimal
    n_compensation: Decimal
    notice_pay: Decimal
    total: Decimal
    formula: str
    legal_basis: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for field in (
            "n_months",
            "original_average_monthly_salary",
            "salary_base",
            "n_compensation",
            "notice_pay",
            "total",
        ):
            result[field] = str(result[field])
        return result


def _money(value: Decimal) -> Decimal:
    return value.quantize(MONEY, rounding=ROUND_HALF_UP)


def calculate_n_months(start_date: date, end_date: date) -> tuple[Decimal, str]:
    """按每满一年1个月、余期满6个月1个月、不满6个月0.5个月计算 N。"""

    if end_date < start_date:
        raise ValueError("解除日期不能早于入职日期")

    delta = relativedelta(end_date, start_date)
    n_months = Decimal(delta.years)
    has_remainder = delta.months > 0 or delta.days > 0
    if delta.months >= 6:
        n_months += Decimal("1")
    elif has_remainder or delta.years == 0:
        n_months += Decimal("0.5")

    period = f"{delta.years}年{delta.months}个月{delta.days}天"
    return n_months, period


def calculate_compensation(
    *,
    start_date: date,
    end_date: date,
    average_monthly_salary: Decimal | int | float | str,
    compensation_type: CompensationType | str,
    notice_monthly_salary: Decimal | int | float | str | None = None,
    local_average_salary: Decimal | int | float | str | None = None,
) -> CompensationResult:
    """计算补偿金额；是否符合相应法律情形需由 Agent 依据案情判断。"""

    kind = CompensationType(compensation_type)
    average_salary = Decimal(str(average_monthly_salary))
    if average_salary <= 0:
        raise ValueError("解除前十二个月平均工资必须大于0")

    if kind is CompensationType.N_PLUS_1 and notice_monthly_salary is None:
        raise MissingCalculationFields(["notice_monthly_salary（解除前上一个月工资）"])

    n_months, service_period = calculate_n_months(start_date, end_date)
    salary_base = average_salary
    warnings: list[str] = []

    if local_average_salary is not None:
        local_salary = Decimal(str(local_average_salary))
        if local_salary <= 0:
            raise ValueError("当地上年度职工月平均工资必须大于0")
        cap = local_salary * Decimal("3")
        if average_salary > cap:
            salary_base = cap
            n_months = min(n_months, Decimal("12"))
            warnings.append("已适用三倍社平工资封顶，补偿年限最高按12年计算。")
    else:
        warnings.append("未提供当地上年度职工月平均工资，结果未校验高收入封顶规则。")

    salary_base = _money(salary_base)
    n_compensation = _money(salary_base * n_months)
    notice_pay = Decimal("0.00")
    legal_basis = ["《中华人民共和国劳动合同法》第四十七条"]

    if kind is CompensationType.N:
        total = n_compensation
        formula = f"{salary_base} × {n_months} = {total}"
        legal_basis.insert(0, "《中华人民共和国劳动合同法》第四十六条")
    elif kind is CompensationType.N_PLUS_1:
        notice_pay = _money(Decimal(str(notice_monthly_salary)))
        if notice_pay <= 0:
            raise ValueError("解除前上一个月工资必须大于0")
        total = _money(n_compensation + notice_pay)
        formula = f"{salary_base} × {n_months} + {notice_pay} = {total}"
        legal_basis.insert(0, "《中华人民共和国劳动合同法》第四十条")
        warnings.append("N+1仅适用于第四十条规定的特定无过失性解除情形。")
    else:
        total = _money(n_compensation * Decimal("2"))
        formula = f"{salary_base} × {n_months} × 2 = {total}"
        legal_basis.append("《中华人民共和国劳动合同法》第八十七条")
        warnings.append("2N以违法解除或违法终止被认定为前提。")

    return CompensationResult(
        compensation_type=kind.value,
        service_period=service_period,
        n_months=n_months,
        original_average_monthly_salary=_money(average_salary),
        salary_base=salary_base,
        n_compensation=n_compensation,
        notice_pay=notice_pay,
        total=total,
        formula=formula,
        legal_basis=legal_basis,
        warnings=warnings,
    )

