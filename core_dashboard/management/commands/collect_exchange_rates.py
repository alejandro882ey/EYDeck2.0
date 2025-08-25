from django.core.management.base import BaseCommand
from core_dashboard.models import ExchangeRate
from core_dashboard.views import get_dolarapi_rates # Re-use the function
from datetime import date

class Command(BaseCommand):
    help = 'Collects daily exchange rates from DolarAPI and saves them to the database.'

    def handle(self, *args, **kwargs):
        self.stdout.write('Attempting to collect daily exchange rates...')
        rates = get_dolarapi_rates()

        if rates and rates.get('oficial') is not None and rates.get('paralelo') is not None:
            try:
                # Get today's date
                today = date.today()

                # Create or update the ExchangeRate entry for today
                exchange_rate_entry, created = ExchangeRate.objects.update_or_create(
                    date=today,
                    defaults={
                        'oficial_rate': rates.get('oficial'),
                        'paralelo_rate': rates.get('paralelo'),
                    }
                )
                if created:
                    self.stdout.write(self.style.SUCCESS(f'Successfully collected and saved new exchange rates for {today}.'))
                else:
                    self.stdout.write(self.style.SUCCESS(f'Successfully updated exchange rates for {today}.'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error saving exchange rates to database: {e}'))
        else:
            self.stdout.write(self.style.WARNING('Failed to retrieve exchange rates from DolarAPI.'))
