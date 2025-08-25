from django.db import models
from django.contrib.auth.models import User

class UploadHistory(models.Model):
    file_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.file_name

class Client(models.Model):
    name = models.CharField(max_length=255)
    # Add other client-related fields as needed

    def __str__(self):
        return self.name

class Contract(models.Model):
    client = models.ForeignKey(Client, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    start_date = models.DateField()
    end_date = models.DateField()
    # Add other contract-related fields as needed

    def __str__(self):
        return self.name

class Area(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class SubArea(models.Model):
    area = models.ForeignKey(Area, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class RevenueEntry(models.Model):
    date = models.DateField()
    client = models.ForeignKey(Client, on_delete=models.CASCADE)
    contract = models.ForeignKey(Contract, on_delete=models.CASCADE, null=True, blank=True)
    area = models.ForeignKey(Area, on_delete=models.CASCADE)
    sub_area = models.ForeignKey(SubArea, on_delete=models.CASCADE, null=True, blank=True)
    revenue = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    engagement_partner = models.CharField(max_length=255, blank=True, null=True)
    engagement_manager = models.CharField(max_length=255, blank=True, null=True)
    collections = models.DecimalField(max_digits=10, decimal_places=2, default=0.00, null=True, blank=True)
    billing = models.DecimalField(max_digits=10, decimal_places=2, default=0.00, null=True, blank=True)
    bcv_rate = models.DecimalField(max_digits=10, decimal_places=4, default=1.0000, null=True, blank=True)
    monitor_rate = models.DecimalField(max_digits=10, decimal_places=4, default=1.0000, null=True, blank=True)

    # New fields from Final_Database.csv
    engagement_id = models.CharField(max_length=255, blank=True, null=True)
    engagement = models.CharField(max_length=255, blank=True, null=True)
    engagement_service_line = models.CharField(max_length=255, blank=True, null=True)
    engagement_sub_service_line = models.CharField(max_length=255, blank=True, null=True)
    fytd_charged_hours = models.FloatField(default=0.0, null=True, blank=True)
    fytd_direct_cost_amt = models.FloatField(default=0.0, null=True, blank=True)
    fytd_ansr_amt = models.FloatField(default=0.0, null=True, blank=True)
    mtd_charged_hours = models.FloatField(default=0.0, null=True, blank=True)
    mtd_direct_cost_amt = models.FloatField(default=0.0, null=True, blank=True)
    mtd_ansr_amt = models.FloatField(default=0.0, null=True, blank=True)
    cp_ansr_amt = models.FloatField(default=0.0, null=True, blank=True)
    duplicate_engagement_id = models.IntegerField(default=0, null=True, blank=True)
    original_week_string = models.CharField(max_length=255, blank=True, null=True) # Original week string from CSV
    periodo_fiscal = models.CharField(max_length=255, blank=True, null=True)
    fecha_cobro = models.CharField(max_length=255, blank=True, null=True) # Keep as CharField due to mixed date/datetime formats
    dif_div = models.FloatField(blank=True, null=True)
    perdida_tipo_cambio_monitor = models.FloatField(blank=True, null=True)
    fytd_diferencial_final = models.FloatField(blank=True, null=True)
    fytd_ansr_sintetico = models.FloatField(blank=True, null=True)
    total_revenue_days_p_cp = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"{self.date} - {self.client.name} - {self.revenue}"

class ExchangeRate(models.Model):
    date = models.DateField(unique=True)
    oficial_rate = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    paralelo_rate = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    differential = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True) # paralelo - oficial

    def save(self, *args, **kwargs):
        if self.paralelo_rate is not None and self.oficial_rate is not None:
            self.differential = self.paralelo_rate - self.oficial_rate
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Exchange Rates for {self.date}: Oficial={self.oficial_rate}, Paralelo={self.paralelo_rate}"