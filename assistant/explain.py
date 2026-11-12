"""Map internal feature names to short phrases a fraud analyst can read."""
import re


def describe(name: str) -> str:
    if name.endswith("_freq"):
        return f"how common the value of {name[:-5]} is"
    if name.startswith("Vblock_"):
        return "a group of Vesta signals is missing for this transaction"
    if name.endswith("_na"):
        return f"{name[:-3]} is missing"
    if "_agg_" in name:
        who = "this client" if name.startswith("uid") else "this card/address"
        kind = "average" if name.endswith("_mean") else "variation"
        col = name.split("_agg_")[1].rsplit("_", 1)[0]
        return f"{kind} of {col} in the past activity of {who}"
    if name.startswith("combo_"):
        return "recent activity of this card + address + email combination (" + name.split("_count_")[1] + ")"
    if name.startswith(("uid_count", "card_count", "card1_count")):
        return "how many transactions this client/card made recently"
    if name.endswith("secs_since_last"):
        return "time since the previous transaction of this client"
    if name.endswith("amt_vs_mean"):
        return "amount compared with what this client usually spends"
    if name.startswith("Dn_"):
        return f"client age signal ({name[3:]} normalized)"
    if re.fullmatch(r"C\d+", name):
        return f"{name}: card-linkage count from Vesta (how the card connects to addresses, emails, devices)"
    if re.fullmatch(r"D\d+", name):
        return f"{name}: time since an earlier event of this client (Vesta)"
    if re.fullmatch(r"M\d+", name):
        return f"{name}: match check between card and billing details"
    if re.fullmatch(r"V\d+", name):
        return f"{name}: Vesta engineered risk signal"
    if name.startswith("na_count"):
        return "how many fields are missing for this transaction"
    if name == "TransactionAmt":
        return "transaction amount"
    return name
