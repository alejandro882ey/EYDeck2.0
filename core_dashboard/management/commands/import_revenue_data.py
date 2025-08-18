import csv
from datetime import datetime
from django.core.management.base import BaseCommand
from core_dashboard.models import Client, Area, SubArea, RevenueEntry, Contract

class Command(BaseCommand):
    help = 'Imports revenue data from Final_Database.csv'

    def handle(self, *args, **options):
        # Adjust this path if your CSV is in a different location
        csv_file_path = '/Users/ericklujan/Downloads/dashboard_django/Final_Database.csv'

        self.stdout.write(self.style.SUCCESS(f'Attempting to import data from {csv_file_path}'))

        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Helper function to get float values, handling empty strings and None
                def get_float_or_none(value):
                    try:
                        return float(value) if value else None
                    except ValueError:
                        return None

                # Helper function to get int values, handling empty strings and None
                def get_int_or_none(value):
                    try:
                        return int(value) if value else None
                    except ValueError:
                        return None

                # Helper function to parse date, handling various formats and errors
                def parse_date(date_string):
                    if not date_string:
                        return None
                    for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d-%m-%Y'):
                        try:
                            return datetime.strptime(date_string, fmt).date()
                        except ValueError:
                            continue
                    return None # Return None if no format matches

                # Create or get Client
                client, _ = Client.objects.get_or_create(name=row['Client'])

                # Create or get Area
                area, _ = Area.objects.get_or_create(name=row['EngagementServiceLine'])

                # Create or get SubArea
                sub_area = None
                if row.get('EngagementSubServiceLine'):
                    sub_area, _ = SubArea.objects.get_or_create(area=area, name=row['EngagementSubServiceLine'])

                # Handle Contract (assuming EngagementID can be a contract identifier)
                contract = None
                if row.get('EngagementID'):
                    contract, _ = Contract.objects.get_or_create(
                        client=client,
                        name=row['Engagement'],
                        defaults={
                            'value': get_float_or_none(row.get('FYTD_ANSRAmt', 0) or 0),
                            'start_date': parse_date(row.get('Week')) or datetime.now().date(),
                            'end_date': parse_date(row.get('Week')) or datetime.now().date(),
                        }
                    )

                # Parse main date field
                entry_date = parse_date(row.get('Week')) or datetime.now().date()

                # Create RevenueEntry
                RevenueEntry.objects.create(
                    date=entry_date,
                    client=client,
                    contract=contract,
                    area=area,
                    sub_area=sub_area,
                    revenue=get_float_or_none(row.get('FYTD_ANSRAmt', 0) or 0),
                    engagement_partner=row.get('EngagementPartner', ''),
                    engagement_manager=row.get('EngagementManager', ''),
                    collections=get_float_or_none(row.get('Collections', 0) or 0),
                    billing=get_float_or_none(row.get('Billing', 0) or 0),
                    bcv_rate=get_float_or_none(row.get('BCV_Rate', 1.0) or 1.0),
                    monitor_rate=get_float_or_none(row.get('Monitor_Rate', 1.0) or 1.0),

                    # New fields from Final_Database.csv
                    engagement_id=row.get('EngagementID', ''),
                    engagement=row.get('Engagement', ''),
                    engagement_service_line=row.get('EngagementServiceLine', ''),
                    engagement_sub_service_line=row.get('EngagementSubServiceLine', ''),
                    fytd_charged_hours=get_float_or_none(row.get('FYTD_ChargedHours', 0) or 0),
                    fytd_direct_cost_amt=get_float_or_none(row.get('FYTD_DirectCostAmt', 0) or 0),
                    fytd_ansr_amt=get_float_or_none(row.get('FYTD_ANSRAmt', 0) or 0),
                    mtd_charged_hours=get_float_or_none(row.get('MTD_ChargedHours', 0) or 0),
                    mtd_direct_cost_amt=get_float_or_none(row.get('MTD_DirectCostAmt', 0) or 0),
                    mtd_ansr_amt=get_float_or_none(row.get('MTD_ANSRAmt', 0) or 0),
                    cp_ansr_amt=get_float_or_none(row.get('CP_ANSRAmt', 0) or 0),
                    duplicate_engagement_id=get_int_or_none(row.get('Duplicate EngagementID', 0) or 0),
                    original_week_string=row.get('Week', ''),
                    periodo_fiscal=row.get('PERIODO FISCAL', ''),
                    fecha_cobro=row.get('Fecha de Cobro', ''),
                    dif_div=get_float_or_none(row.get('Dif_Div')),
                    perdida_tipo_cambio_monitor=get_float_or_none(row.get('Perdida al tipo de cambio Monitor')),
                    fytd_diferencial_final=get_float_or_none(row.get('FYTD_diferencial_final')),
                    fytd_ansr_sintetico=get_float_or_none(row.get('FYTD_ANSR_Sintetico')),
                )
        self.stdout.write(self.style.SUCCESS('Successfully imported data from Final_Database.csv'))