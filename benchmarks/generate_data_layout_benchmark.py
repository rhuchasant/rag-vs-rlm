"""
Generate synthetic multi-tab data layout benchmark.
Simulates the real-world Excel-with-19-tabs scenario.
Supports scaling to ~750k chars to match production POC size.
"""

import json
import random
import string
from pathlib import Path

# Base templates (19). Extra templates for scaling to 30+ tabs.
TAB_TEMPLATES = [
    ("Customer_Data", "customer_export.csv", ["customer_id", "name", "email", "created_at", "region"]),
    ("Order_Data", "orders_export.csv", ["order_id", "customer_id", "order_date", "total_amount", "status"]),
    ("Product_Catalog", "products.csv", ["product_id", "product_name", "category", "unit_price", "stock_quantity"]),
    ("Inventory_Log", "inventory_log.csv", ["log_id", "product_id", "warehouse_id", "quantity_change", "timestamp"]),
    ("Shipping_Info", "shipping.csv", ["shipment_id", "order_id", "carrier", "tracking_number", "estimated_delivery"]),
    ("Payment_Records", "payments.csv", ["payment_id", "order_id", "amount", "method", "transaction_date"]),
    ("Returns_Data", "returns.csv", ["return_id", "order_id", "product_id", "reason", "return_date"]),
    ("Warehouse_Locations", "warehouses.csv", ["warehouse_id", "location_name", "address", "capacity"]),
    ("Employee_Data", "employees.csv", ["employee_id", "name", "department", "hire_date", "salary_band"]),
    ("Supplier_Info", "suppliers.csv", ["supplier_id", "company_name", "contact", "lead_time_days"]),
    ("Promotions", "promotions.csv", ["promo_id", "name", "discount_pct", "start_date", "end_date"]),
    ("Audit_Log", "audit_log.csv", ["log_id", "table_name", "action", "user_id", "timestamp"]),
    ("User_Sessions", "sessions.csv", ["session_id", "user_id", "start_time", "end_time", "device"]),
    ("Feedback_Data", "feedback.csv", ["feedback_id", "order_id", "rating", "comment", "submitted_at"]),
    ("Campaign_Tracking", "campaigns.csv", ["campaign_id", "name", "channel", "budget", "conversions"]),
    ("Tax_Rates", "tax_rates.csv", ["region", "rate_pct", "effective_from", "effective_to"]),
    ("Currency_Rates", "fx_rates.csv", ["currency_pair", "rate", "as_of_date"]),
    ("Config_Settings", "config.csv", ["setting_key", "setting_value", "environment"]),
    ("API_Logs", "api_logs.csv", ["request_id", "endpoint", "status_code", "latency_ms", "timestamp"]),
]

# Extra templates for scaling to 30+ tabs
EXTRA_TAB_TEMPLATES = [
    ("Report_Q1", "report_q1.csv", ["report_id", "period", "metric", "value", "updated_at"]),
    ("Report_Q2", "report_q2.csv", ["report_id", "period", "metric", "value", "updated_at"]),
    ("Analytics_Dashboard", "analytics.csv", ["dashboard_id", "widget_id", "config", "refresh_interval"]),
    ("Staging_Data", "staging.csv", ["batch_id", "source_file", "rows_loaded", "status", "loaded_at"]),
    ("Archive_Log", "archive_log.csv", ["archive_id", "table_name", "archived_at", "row_count"]),
    ("Ref_Data_Countries", "ref_countries.csv", ["country_code", "name", "region", "currency"]),
    ("Ref_Data_Currencies", "ref_currencies.csv", ["code", "name", "symbol", "decimal_places"]),
    ("Integration_Events", "integration_events.csv", ["event_id", "source", "payload", "processed_at"]),
    ("Error_Log", "error_log.csv", ["error_id", "message", "stack_trace", "occurred_at"]),
    ("Metrics_Daily", "metrics_daily.csv", ["date", "metric_name", "value", "dimension"]),
    ("Metrics_Hourly", "metrics_hourly.csv", ["ts", "metric_name", "value", "dimension"]),
    ("User_Preferences", "user_prefs.csv", ["user_id", "pref_key", "pref_value", "updated_at"]),
    ("Notification_Queue", "notifications.csv", ["notif_id", "user_id", "channel", "sent_at"]),
    ("Cache_Stats", "cache_stats.csv", ["cache_key", "hits", "misses", "last_updated"]),
    ("Backup_Registry", "backups.csv", ["backup_id", "path", "size_bytes", "created_at"]),
]

TYPES = ["INT", "VARCHAR", "DECIMAL", "DATE", "TIMESTAMP", "TEXT", "BOOLEAN"]

# Extra column names for expanding layouts (realistic data warehouse fields)
EXTRA_COLUMNS = [
    "id", "uuid", "version", "created_by", "updated_by", "created_at", "updated_at",
    "deleted_at", "tenant_id", "source_system", "batch_id", "checksum", "row_number",
    "effective_from", "effective_to", "is_current", "metadata", "raw_payload",
    "status", "error_code", "retry_count", "processed_at", "export_date",
]

DATA_QUALITY_RULES = [
    "NotNull: Field must not contain null values",
    "FormatCheck: Value must match expected format pattern",
    "RangeCheck: Numeric values must be within valid range",
    "ReferentialIntegrity: FK must reference existing parent record",
    "Uniqueness: Combination of fields must be unique",
    "DefaultValue: When null, apply default from config",
]


def generate_tab_content(
    tab_name: str,
    filename: str,
    columns: list,
    target_chars_per_tab: int = 500,
) -> str:
    """Generate a single tab's data layout specification. Expands to target size."""
    lines = [
        f"# {tab_name}",
        f"File: {filename}",
        f"Description: Data layout specification for {filename}. This tab defines the schema, constraints, and metadata.",
        "",
        "## Column Definitions",
        "Columns:",
    ]
    for col in columns:
        col_type = random.choice(TYPES)
        if col_type == "VARCHAR":
            size = random.choice([50, 100, 255, 500])
            lines.append(f"- {col} ({col_type} {size})")
        elif col_type == "DECIMAL":
            lines.append(f"- {col} ({col_type} 10,2)")
        else:
            lines.append(f"- {col} ({col_type})")

    current_len = len("\n".join(lines))
    chars_needed = target_chars_per_tab - current_len

    if chars_needed > 200:
        # Add extra columns to reach target size
        num_extra = min(400, chars_needed // 50)
        lines.append("")
        lines.append("## Additional Schema Fields")
        for i in range(num_extra):
            base = random.choice(EXTRA_COLUMNS)
            col = f"{base}_{i}" if i > 5 else base
            col_type = random.choice(TYPES)
            if col_type == "VARCHAR":
                lines.append(f"- {col} ({col_type} {random.choice([50, 100, 255])})")
            else:
                lines.append(f"- {col} ({col_type})")

        # Add constraints, indexes
        lines.append("")
        lines.append("## Constraints and Indexes")
        lines.append(f"- Primary key: {columns[0]}")
        lines.append("- Unique constraint on (tenant_id, batch_id) where applicable")
        lines.append("- Foreign key references to parent tables")
        lines.append("")
        lines.append("## Data Quality Rules")
        for _ in range(min(30, chars_needed // 60)):
            rule = random.choice(DATA_QUALITY_RULES)
            extra = f" - Applies to {random.choice(columns)}" if columns else ""
            lines.append(f"- {rule}{extra}")

        # Pad with documentation block to reach target (realistic verbose spec)
        remaining = target_chars_per_tab - len("\n".join(lines))
        if remaining > 200:
            lines.append("")
            lines.append("## Technical Documentation")
            block_size = 150
            num_blocks = max(1, remaining // block_size)
            doc_templates = [
                "Field {col} is used for tracking changes in the export process. This column is populated by the ETL job during nightly batch runs.",
                "Index on {col} improves query performance for batch operations. Consider composite index with tenant_id for multi-tenant queries.",
                "Constraint: {col} must be populated when status is COMPLETED. Validation runs before data is committed to the warehouse.",
                "The {col} field stores metadata from the source system. Format may vary by source; see mapping documentation.",
            ]
            for i in range(num_blocks):
                t = doc_templates[i % len(doc_templates)]
                col = random.choice(columns) if columns else "id"
                lines.append(f"- {t.format(col=col)}")

    return "\n".join(lines)


def _get_all_templates() -> list:
    """Return combined templates for scaling beyond 19 tabs."""
    return list(TAB_TEMPLATES) + list(EXTRA_TAB_TEMPLATES)


def generate_benchmark(
    num_tabs: int = 19,
    output_dir: Path = None,
    target_total_chars: int = 750000,
    seed: int = 42,
) -> dict:
    """Generate benchmark with N tabs. Scales to target_total_chars (~750k for POC match)."""
    random.seed(seed)
    if output_dir is None:
        output_dir = Path(__file__).parent

    all_templates = _get_all_templates()
    if num_tabs > len(all_templates):
        raise ValueError(
            f"num_tabs={num_tabs} exceeds available templates ({len(all_templates)}). "
            f"Add more to EXTRA_TAB_TEMPLATES or use num_tabs <= {len(all_templates)}."
        )

    chars_per_tab = max(500, target_total_chars // num_tabs)
    tab_contents = {}
    full_text_parts = []

    for i in range(num_tabs):
        name, filename, cols = all_templates[i]
        content = generate_tab_content(name, filename, cols, target_chars_per_tab=chars_per_tab)
        tab_contents[f"Tab_{i+1}_{name}"] = content
        full_text_parts.append(f"\n\n## {name}\n{content}")

    full_text = "# Multi-Tab Data Layout Specification\n" + "\n".join(full_text_parts)

    # Save as JSON (for RLM - dict structure)
    json_path = output_dir / "data_layout_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(tab_contents, f, indent=2)

    # Save as text (for RAG - concatenated)
    txt_path = output_dir / "data_layout_benchmark.txt"
    with open(txt_path, "w") as f:
        f.write(full_text)

    return {"tabs": tab_contents, "text": full_text}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabs", type=int, default=30, help="Number of tabs (max 34 with current templates)")
    parser.add_argument("--size", type=int, default=750000, help="Target total chars (~189k tokens at 4 chars/token)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    benchmark = generate_benchmark(args.tabs, target_total_chars=args.size, seed=args.seed)
    print(f"Generated benchmark with {len(benchmark['tabs'])} tabs")
    print(f"Total chars: {len(benchmark['text']):,} (~{len(benchmark['text'])//4:,} tokens)")
