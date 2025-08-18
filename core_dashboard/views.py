# from django.contrib.auth.models import User

# # Create users if they don't exist
# if not User.objects.filter(username='admin').exists():
#     User.objects.create_user('admin', password='admin')
# if not User.objects.filter(username='dev').exists():
#     User.objects.create_user('dev', password='dev')

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm

def custom_login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                if username == 'dev':
                    return redirect('upload_file')
                else:
                    return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'registration/login.html', {'form': form})

def custom_logout_view(request):
    logout(request)
    return redirect('login')


from core_dashboard.models import RevenueEntry, Client, Area, SubArea, Contract
from django.db.models import Sum, Count, Q, F
from django.db import models
from django.db.models.functions import TruncWeek
from django.utils import timezone
import datetime
import json
import requests # Import the requests library
from django.http import JsonResponse # Import JsonResponse

def get_dolarapi_rates():
    try:
        response = requests.get("https://ve.dolarapi.com/v1/dolares")
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        oficial_rate = None
        paralelo_rate = None
        for rate_entry in data:
            if rate_entry.get("fuente") == "oficial":
                oficial_rate = rate_entry.get("promedio")
            elif rate_entry.get("fuente") == "paralelo":
                paralelo_rate = rate_entry.get("promedio")
        print(f"DolarAPI Raw Data: {data}") # Debug print
        print(f"Oficial Rate Found: {oficial_rate}") # Debug print
        print(f"Paralelo Rate Found: {paralelo_rate}") # Debug print
        return oficial_rate, paralelo_rate
    except requests.exceptions.RequestException as e:
        print(f"Error fetching DolarAPI rates: {e}")
        return None, None

# New API view for exchange rates
def exchange_rates_api(request):
    oficial_rate, paralelo_rate = get_dolarapi_rates()
    data = {
        'oficial_rate': oficial_rate,
        'paralelo_rate': paralelo_rate,
    }
    return JsonResponse(data)

from django.contrib.auth.decorators import login_required, user_passes_test

def is_dev_user(user):
    return user.username == 'dev'

from .data_processor import process_uploaded_files
from .models import UploadHistory
from django.core.files.storage import FileSystemStorage
import pandas as pd
from django.conf import settings
import os
import traceback
import datetime
import subprocess # Import subprocess

ALLOWED_EXTENSIONS = {'.csv', '.xls', '.xlsx', '.xlsb'}

@login_required
@user_passes_test(is_dev_user)
def upload_file_view(request):
    history = UploadHistory.objects.all().order_by('-uploaded_at')
    if request.method == 'POST':
        try:
            upload_date_str = request.POST.get('upload_date')
            if not upload_date_str:
                raise ValueError("Upload date is required.")
            upload_date = datetime.datetime.strptime(upload_date_str, '%Y-%m-%d').date()

            engagement_file = request.FILES.get('engagement_df_file')
            dif_file = request.FILES.get('dif_df_file')
            revenue_days_file = request.FILES.get('revenue_days_file')

            if not all([engagement_file, dif_file, revenue_days_file]):
                raise ValueError("All three files are required.")

            # Validate file extensions
            for f in [engagement_file, dif_file, revenue_days_file]:
                ext = os.path.splitext(f.name)[1]
                if ext.lower() not in ALLOWED_EXTENSIONS:
                    raise ValueError(f"File type {ext} is not allowed.")

            # Create history directory for the given date
            history_dir = os.path.join(settings.MEDIA_ROOT, 'historico_de_final_database', upload_date.strftime('%Y-%m-%d'))
            os.makedirs(history_dir, exist_ok=True)

            fs = FileSystemStorage(location=history_dir)

            # Save files with new names
            engagement_ext = os.path.splitext(engagement_file.name)[1]
            dif_ext = os.path.splitext(dif_file.name)[1]
            revenue_ext = os.path.splitext(revenue_days_file.name)[1]

            engagement_filename = fs.save(f"Engagement_df_{upload_date_str}{engagement_ext}", engagement_file)
            dif_filename = fs.save(f"Dif_df_{upload_date_str}{dif_ext}", dif_file)
            revenue_filename = fs.save(f"Revenue_days_{upload_date_str}{revenue_ext}", revenue_days_file)

            engagement_path = fs.path(engagement_filename)
            dif_path = fs.path(dif_filename)
            revenue_path = fs.path(revenue_filename)

            # --- NEW: Call process_uploaded_data.py as a subprocess ---
            process_script_path = os.path.join(settings.BASE_DIR, 'process_uploaded_data.py')
            command = [
                'python',
                process_script_path,
                engagement_path,
                dif_path,
                revenue_path,
                upload_date_str
            ]
            print(f"Executing command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            print(f"Subprocess Return Code: {result.returncode}")
            print(f"Subprocess STDOUT: {result.stdout}")
            print(f"Subprocess STDERR: {result.stderr}")

            if result.returncode != 0:
                error_message = f"Error processing files: {result.stderr}"
                print(f"Subprocess Error: {error_message}")
                raise Exception(error_message)

            # --- END NEW ---

            # Read the newly generated Final_Database.csv
            processed_data_filename = f"Final_Database_{upload_date_str}.csv"
            processed_data_path = os.path.join(history_dir, processed_data_filename)

            if not os.path.exists(processed_data_path):
                raise FileNotFoundError(f"Processed file not found: {processed_data_path}. Check process_uploaded_data.py output for errors.")

            final_database = pd.read_csv(processed_data_path)

            # Record the upload in history
            UploadHistory.objects.create(
                file_name=f"Engagement_df_{upload_date_str}, Dif_df_{upload_date_str}, Revenue_days_{upload_date_str}",
                uploaded_by=request.user
            )

            df_html = final_database.head(10).to_html(classes='table table-dark table-striped table-hover', index=False)
            context = {'history': history, 'df_html': df_html, 'success_message': 'Files uploaded and processed successfully!'}
            return render(request, 'core_dashboard/upload.html', context)

        except Exception as e:
            print(f"Error during file upload or processing: {e}")
            traceback.print_exc()
            context = {'history': history, 'error_message': f'Error: {e}'}
            return render(request, 'core_dashboard/upload.html', context)

    return render(request, 'core_dashboard/upload.html', {'history': history})


@login_required
def tables_view(request):
    return render(request, 'core_dashboard/tables.html')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# Import the new analytics engine
from ey_analytics_engine import fetch_all_data, generate_dashboard_analytics
import plotly.graph_objects as go

@login_required
def analysis_view(request):
    try:
        # 1. Fetch the data using the new engine
        master_df = fetch_all_data(historical_csv_path='historical_data.csv')
        print(f"Length of master_df: {len(master_df)}")

        # 2. Generate the analytics and charts
        if len(master_df) > 20:
            final_dashboard_data = generate_dashboard_analytics(master_df)

            # 3. Process the output for the template
            # Trends
            trends_html = ""
            if 'moving_averages_chart' in final_dashboard_data['Trends'] and isinstance(final_dashboard_data['Trends']['moving_averages_chart'], go.Figure):
                trends_html += final_dashboard_data['Trends']['moving_averages_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                trends_html += "<p>Moving Averages chart not available.</p>"

            if 'hp_filter_chart' in final_dashboard_data['Trends'] and isinstance(final_dashboard_data['Trends']['hp_filter_chart'], go.Figure):
                trends_html += final_dashboard_data['Trends']['hp_filter_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                trends_html += "<p>HP Filter chart not available.</p>"

            if 'garch_volatility_chart' in final_dashboard_data['Trends'] and isinstance(final_dashboard_data['Trends']['garch_volatility_chart'], go.Figure):
                trends_html += final_dashboard_data['Trends']['garch_volatility_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                trends_html += "<p>GARCH Volatility chart not available.</p>"

            if 'latest_volatility' in final_dashboard_data['Trends']:
                trends_html += f"<h5>Latest Volatility: {final_dashboard_data['Trends']['latest_volatility']:.4f}</h5>"

            # Projections
            projections_html = ""
            if 'arima_forecast_chart' in final_dashboard_data['Projections'] and isinstance(final_dashboard_data['Projections']['arima_forecast_chart'], go.Figure):
                projections_html += final_dashboard_data['Projections']['arima_forecast_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                projections_html += "<p>ARIMA Forecast chart not available.</p>"

            if 'holt_winters_chart' in final_dashboard_data['Projections'] and isinstance(final_dashboard_data['Projections']['holt_winters_chart'], go.Figure):
                projections_html += final_dashboard_data['Projections']['holt_winters_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                projections_html += "<p>Holt-Winters Forecast chart not available.</p>"

            # Estimates
            estimations_html = ""
            if 'spread_chart' in final_dashboard_data['Estimates'] and isinstance(final_dashboard_data['Estimates']['spread_chart'], go.Figure):
                estimations_html += final_dashboard_data['Estimates']['spread_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                estimations_html += "<p>Spread chart not available.</p>"

            if 'var_irf_chart' in final_dashboard_data['Estimates'] and isinstance(final_dashboard_data['Estimates']['var_irf_chart'], go.Figure):
                estimations_html += final_dashboard_data['Estimates']['var_irf_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                estimations_html += "<p>VAR IRF chart not available.</p>"

            if 'latest_spread' in final_dashboard_data['Estimates']:
                estimations_html += f"<h5>Latest Spread: {final_dashboard_data['Estimates']['latest_spread']:.2f}%</h5>"

            # Benchmarking
            benchmarking_html = ""
            if 'benchmark_chart' in final_dashboard_data['Benchmarking'] and isinstance(final_dashboard_data['Benchmarking']['benchmark_chart'], go.Figure):
                benchmarking_html += final_dashboard_data['Benchmarking']['benchmark_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                benchmarking_html += "<p>Benchmark chart not available.</p>"

            if 'forecast_surprise' in final_dashboard_data['Benchmarking']:
                benchmarking_html += f"<h5>Forecast Surprise: {final_dashboard_data['Benchmarking']['forecast_surprise']:.4f}</h5>"

            # Competitive Landscape
            competitive_landscape_html = ""
            if 'share_of_voice_chart' in final_dashboard_data['Competitive_Landscape'] and isinstance(final_dashboard_data['Competitive_Landscape']['share_of_voice_chart'], go.Figure):
                competitive_landscape_html += final_dashboard_data['Competitive_Landscape']['share_of_voice_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                competitive_landscape_html += "<p>Share of Voice chart not available.</p>"

            if 'brand_interest_chart' in final_dashboard_data['Competitive_Landscape'] and isinstance(final_dashboard_data['Competitive_Landscape']['brand_interest_chart'], go.Figure):
                competitive_landscape_html += final_dashboard_data['Competitive_Landscape']['brand_interest_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                competitive_landscape_html += "<p>Brand Interest chart not available.</p>"

            if 'talent_acquisition_chart' in final_dashboard_data['Competitive_Landscape'] and isinstance(final_dashboard_data['Competitive_Landscape']['talent_acquisition_chart'], go.Figure):
                competitive_landscape_html += final_dashboard_data['Competitive_Landscape']['talent_acquisition_chart'].to_html(full_html=False, include_plotlyjs='cdn')
            else:
                competitive_landscape_html += "<p>Talent Acquisition chart not available.</p>"

            context = {
                'trends': trends_html,
                'projections': projections_html,
                'estimations': estimations_html,
                'expected_data': benchmarking_html,
                'competitive_landscape': competitive_landscape_html, # Added new context variable
            }
        else:
            context = {
                'trends': "<p>Not enough historical data for analysis. Need at least 20 data points.</p>",
                'projections': "",
                'estimations': "",
                'expected_data': "",
                'competitive_landscape': "", # Added new context variable
            }
        print(f"Context keys: {context.keys()}")
        for key, value in context.items():
            if isinstance(value, str):
                print(f"Context['{key}'] length: {len(value)}")
                if len(value) < 500: # Print short values for inspection
                    print(f"Context['{key}'] content: {value[:200]}...")
            else:
                print(f"Context['{key}'] type: {type(value)}")

    except Exception as e:
        print(f"Error in analysis_view: {e}")
        traceback.print_exc()
        context = {
            'trends': f"<p>An error occurred during analysis: {e}</p>",
            'projections': "",
            'estimations': "",
            'expected_data': "",
            'competitive_landscape': "", # Added new context variable
        }

    return render(request, 'core_dashboard/analysis.html', context)

@login_required
def messaging_view(request):
    # Get distinct sub-areas for messaging
    sub_areas = SubArea.objects.values_list('name', flat=True).distinct().exclude(name__isnull=True).exclude(name__exact='').order_by('name')
    context = {'sub_areas': sub_areas}
    return render(request, 'core_dashboard/messaging.html', context)

@login_required
def dashboard_view(request):
    # Get filter parameters from request
    selected_partner = request.GET.get('partner')
    selected_manager = request.GET.get('manager')
    selected_area = request.GET.get('area')
    selected_sub_area = request.GET.get('sub_area')
    selected_client = request.GET.get('client')
    selected_week_filter = request.GET.get('week') # Renamed to avoid conflict

    # Get the most recent date from all RevenueEntry objects
    latest_entry = RevenueEntry.objects.all().order_by('-date').first()
    latest_date = latest_entry.date if latest_entry else None

    # Queryset for historical trend (uses all data)
    all_revenue_entries = RevenueEntry.objects.all()

    # Base queryset for KPIs. Start with all entries.
    revenue_entries_for_kpis = RevenueEntry.objects.all()

    # Apply user filters to KPI-specific queryset
    if selected_partner:
        revenue_entries_for_kpis = revenue_entries_for_kpis.filter(engagement_partner=selected_partner)
    if selected_manager:
        revenue_entries_for_kpis = revenue_entries_for_kpis.filter(engagement_manager=selected_manager)
    if selected_area:
        revenue_entries_for_kpis = revenue_entries_for_kpis.filter(area__name=selected_area)
    if selected_sub_area:
        revenue_entries_for_kpis = revenue_entries_for_kpis.filter(sub_area__name=selected_sub_area)
    if selected_client:
        revenue_entries_for_kpis = revenue_entries_for_kpis.filter(client__name=selected_client)

    # Date filtering logic
    if selected_week_filter:
        # The filter gives us the Friday of the week.
        friday_date = datetime.datetime.strptime(selected_week_filter, '%Y-%m-%d').date()
        # weekday() returns 0 for Monday and 6 for Sunday. Friday is 4.
        start_of_week = friday_date - datetime.timedelta(days=friday_date.weekday())
        end_of_week = start_of_week + datetime.timedelta(days=6)
        revenue_entries_for_kpis = revenue_entries_for_kpis.filter(date__range=[start_of_week, end_of_week])
    else:
        # If no week is selected, default to the latest date with data.
        if latest_date:
            revenue_entries_for_kpis = revenue_entries_for_kpis.filter(date=latest_date)
        else:
            revenue_entries_for_kpis = RevenueEntry.objects.none() # No data at all

    # KPIs
    ansr_sintetico = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(Sum('revenue'))['revenue__sum'] or 0)
    total_clients = "{:,.0f}".format(revenue_entries_for_kpis.values('client').distinct().count())
    total_engagements = "{:,.0f}".format(revenue_entries_for_kpis.values('contract').distinct().count())

    # --- Macro Section Calculations ---
    macro_total_clients = revenue_entries_for_kpis.values('client').distinct().count()
    macro_total_ansr_sintetico = revenue_entries_for_kpis.aggregate(Sum('fytd_ansr_sintetico'))['fytd_ansr_sintetico__sum'] or 0
    macro_total_direct_cost = revenue_entries_for_kpis.aggregate(Sum('fytd_direct_cost_amt'))['fytd_direct_cost_amt__sum'] or 0
    macro_margin = macro_total_ansr_sintetico - macro_total_direct_cost
    macro_margin_percentage = (macro_margin / macro_total_ansr_sintetico * 100) if macro_total_ansr_sintetico else 0
    macro_total_charged_hours = revenue_entries_for_kpis.aggregate(Sum('fytd_charged_hours'))['fytd_charged_hours__sum'] or 0
    macro_rph = (macro_total_ansr_sintetico / macro_total_charged_hours) if macro_total_charged_hours else 0
    macro_mtd_charged_hours = revenue_entries_for_kpis.aggregate(Sum('mtd_charged_hours'))['mtd_charged_hours__sum'] or 0
    macro_monthly_tracker = macro_mtd_charged_hours * macro_rph

    # Calculate FYTD, MTD, and Daily Revenue
    today = timezone.now().date()
    current_month_start = today.replace(day=1)
    current_year_start = today.replace(month=1, day=1) # Assuming fiscal year starts Jan 1

    fytd_revenue = "${:,.2f}".format(revenue_entries_for_kpis.filter(date__gte=current_year_start).aggregate(Sum('revenue'))['revenue__sum'] or 0)
    mtd_revenue = "${:,.2f}".format(revenue_entries_for_kpis.filter(date__gte=current_month_start).aggregate(Sum('revenue'))['revenue__sum'] or 0)
    daily_revenue = "${:,.2f}".format(revenue_entries_for_kpis.filter(date=today).aggregate(Sum('revenue'))['revenue__sum'] or 0)

    # Placeholder for Collections and Billing (assuming fields exist in RevenueEntry)
    total_collections = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(Sum('collections'))['collections__sum'] or 0)
    total_billing = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(Sum('billing'))['billing__sum'] or 0)

    # Placeholder for Active Employees in Venezuela
    active_employees_venezuela = "{:,.0f}".format(150) # Static placeholder value

    # Top Partners by Revenue
    top_partners = revenue_entries_for_kpis.values('engagement_partner').annotate(
        total_revenue=Sum('revenue')
    ).order_by('-total_revenue').exclude(engagement_partner__isnull=True).exclude(engagement_partner__exact='')[:5]
    # Format revenue for top_partners
    for p in top_partners:
        p['total_revenue'] = "${:,.2f}".format(p['total_revenue'])

    # Top 5 Clients by Revenue (for table display)
    top_clients_table = revenue_entries_for_kpis.values('client__name').annotate(total_revenue=Sum('revenue')).order_by('-total_revenue')[:5]
    # Format revenue for top_clients_table
    for c in top_clients_table:
        c['total_revenue'] = "${:,.2f}".format(c['total_revenue'])

    # Calculate "Loss per differential"
    # Assuming 'bcv_rate' and 'monitor_rate' are fields in RevenueEntry
    # Loss = (BCV Rate - Monitor Rate) * Revenue
    loss_per_differential = "${:,.2f}".format(revenue_entries_for_kpis.annotate(
        differential_loss=(F('bcv_rate') - F('monitor_rate')) * F('revenue')
    ).aggregate(Sum('differential_loss'))['differential_loss__sum'] or 0)

    # Revenue by Area
    revenue_by_area = revenue_entries_for_kpis.values('area__name').annotate(total_revenue=Sum('revenue')).order_by('-total_revenue')
    area_labels = [item['area__name'] for item in revenue_by_area]
    area_data = [float(item['total_revenue']) for item in revenue_by_area]

    # Top 5 Clients by Revenue (for chart)
    top_clients_chart = revenue_entries_for_kpis.values('client__name').annotate(total_revenue=Sum('revenue')).order_by('-total_revenue')[:5]
    client_labels = [item['client__name'] for item in top_clients_chart]
    client_data = [float(item['total_revenue']) for item in top_clients_chart]

    # Revenue Trend by Date (calculating daily revenue in Python)
    import pandas as pd

    all_entries = list(all_revenue_entries.order_by('engagement_id', 'date').values('engagement_id', 'date', 'revenue'))

    daily_revenues = []
    prev_engagement_id = None
    prev_revenue = 0

    for entry in all_entries:
        current_engagement_id = entry['engagement_id']
        current_revenue = entry['revenue'] or 0

        if current_engagement_id != prev_engagement_id:
            # New engagement, so the daily revenue is the current revenue
            daily_revenue = current_revenue
        else:
            # Same engagement, calculate the difference
            daily_revenue = current_revenue - prev_revenue

        daily_revenues.append({
            'date': entry['date'],
            'daily_revenue': daily_revenue
        })

        prev_engagement_id = current_engagement_id
        prev_revenue = current_revenue

    # Sum up daily revenues by date
    df = pd.DataFrame(daily_revenues)
    if not df.empty:
        # Ensure 'daily_revenue' is numeric before summing
        df['daily_revenue'] = pd.to_numeric(df['daily_revenue'], errors='coerce').fillna(0)
        daily_totals = df.groupby('date')['daily_revenue'].sum().reset_index()
        # Ensure the 'date' column is in datetime format before using .dt accessor
        daily_totals['date'] = pd.to_datetime(daily_totals['date'])
        trend_labels = daily_totals['date'].dt.strftime('%Y-%m-%d').tolist()
        trend_data = [float(x) for x in daily_totals['daily_revenue']]
    else:
        trend_labels = []
        trend_data = []

    # Recent Revenue Entries (uses all_revenue_entries, but limited to 10)
    recent_entries = all_revenue_entries.select_related('client', 'area').order_by('-date')[:10] # Get last 10 entries
    # Format revenue for recent_entries
    for entry in recent_entries:
        entry.revenue_formatted = "${:,.2f}".format(entry.revenue)

    # Get distinct values for filters, excluding None and empty strings
    partners = RevenueEntry.objects.values_list('engagement_partner', flat=True).distinct().exclude(engagement_partner__isnull=True).exclude(Q(engagement_partner__exact='')).order_by('engagement_partner')
    managers = RevenueEntry.objects.values_list('engagement_manager', flat=True).distinct().exclude(engagement_manager__isnull=True).exclude(Q(engagement_manager__exact='')).order_by('engagement_manager')
    areas = Area.objects.values_list('name', flat=True).distinct().exclude(name__isnull=True).exclude(Q(name__exact='')).order_by('name')
    sub_areas = SubArea.objects.values_list('name', flat=True).distinct().exclude(name__isnull=True).exclude(Q(name__exact='')).order_by('name')
    clients = Client.objects.values_list('name', flat=True).distinct().exclude(name__isnull=True).exclude(Q(name__exact='')).order_by('name')

    # Get distinct weeks for filtering, formatted to Friday's date
    available_weeks_raw = RevenueEntry.objects.annotate(calculated_week=TruncWeek('date')).values_list('calculated_week', flat=True).distinct().order_by('calculated_week')
    available_weeks = []
    for week_start_date in available_weeks_raw:
        if week_start_date:
            # Calculate Friday's date for the week (assuming week starts on Monday)
            # Monday (0) to Sunday (6). Friday is 4.
            # If week_start_date is Monday, add 4 days to get Friday.
            friday_date = week_start_date + datetime.timedelta(days=4)
            available_weeks.append(friday_date.strftime('%Y-%m-%d'))

    # Placeholder for highlights/news ticker
    highlights = [
        "EY Global: Janet Truncale elected Global Chair and CEO, effective July 1, 2024.",
        "EY US acquired IT consulting firm Nuvalence, expanding its tech capabilities.",
        "EY Venezuela: Achieved record revenue in Q2 2025, driven by new client acquisitions.",
        "EY Global: New solutions for risk management launched on EY.ai Agentic Platform.",
        "EY Venezuela: Successfully completed major audit for a leading financial institution.",
        "EY Global: EY and ACCA issue new guidance urging stronger AI checks.",
    ]

    # --- Macro Section Calculations ---
    macro_total_clients = revenue_entries_for_kpis.values('client').distinct().count()
    macro_total_ansr_sintetico = revenue_entries_for_kpis.aggregate(Sum('fytd_ansr_sintetico'))['fytd_ansr_sintetico__sum'] or 0
    macro_total_direct_cost = revenue_entries_for_kpis.aggregate(Sum('fytd_direct_cost_amt'))['fytd_direct_cost_amt__sum'] or 0
    macro_margin = macro_total_ansr_sintetico - macro_total_direct_cost
    macro_margin_percentage = (macro_margin / macro_total_ansr_sintetico * 100) if macro_total_ansr_sintetico else 0
    macro_total_charged_hours = revenue_entries_for_kpis.aggregate(Sum('fytd_charged_hours'))['fytd_charged_hours__sum'] or 0
    macro_rph = (macro_total_ansr_sintetico / macro_total_charged_hours) if macro_total_charged_hours else 0
    macro_mtd_charged_hours = revenue_entries_for_kpis.aggregate(Sum('mtd_charged_hours'))['mtd_charged_hours__sum'] or 0
    macro_monthly_tracker = macro_mtd_charged_hours * macro_rph


    # --- Nuevas métricas y datos para gráficos ---

    # 1. Distribución de clientes por partner
    clients_by_partner = revenue_entries_for_kpis.values('engagement_partner').annotate(
        num_clients=Count('client', distinct=True)
    ).order_by('-num_clients').exclude(engagement_partner__isnull=True).exclude(engagement_partner__exact='')

    partner_distribution_labels = [item['engagement_partner'] for item in clients_by_partner]
    partner_distribution_data = [item['num_clients'] for item in clients_by_partner]

    # 2. Cartera en moneda extranjera (usando fytd_diferencial_final)
    cartera_moneda_extranjera = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(Sum('fytd_diferencial_final'))['fytd_diferencial_final__sum'] or 0)

    # 3. Cartera local ajustada (usando fytd_ansr_sintetico)
    cartera_local_ajustada = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(Sum('fytd_ansr_sintetico'))['fytd_ansr_sintetico__sum'] or 0)

    # 4. Total CXC (asumiendo que es la suma de billing - collections, o solo billing si no hay un campo de CXC explícito)
    # Si tienes un campo de CXC directo, por favor, indícalo.
    total_cxc = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(Sum('billing'))['billing__sum'] or 0) # Usando billing como proxy

    # 5. Promedio de antigüedad (requiere un campo de fecha de inicio de contrato/cliente y fecha actual)
    # Por ahora, es un placeholder. Necesito más información sobre cómo calcularlo.
    promedio_antiguedad = "N/A" # Placeholder

    # 6. Unbilled Inventory por Service Line
    # Asumiendo que Unbilled Inventory = FYTD_ANSRAmt - Collections
    unbilled_inventory_by_service_line = revenue_entries_for_kpis.values('engagement_service_line').annotate(
        unbilled_amount=Sum(F('fytd_ansr_amt') - F('collections'), output_field=models.FloatField())
    ).order_by('-unbilled_amount').exclude(engagement_service_line__isnull=True).exclude(engagement_service_line__exact='')

    unbilled_labels = [item['engagement_service_line'] for item in unbilled_inventory_by_service_line]
    unbilled_data = [float(item['unbilled_amount']) if item['unbilled_amount'] is not None else 0.0 for item in unbilled_inventory_by_service_line]

    # 7. Anticipos (requiere un campo específico para anticipos)
    # Por ahora, es un placeholder. Necesito más información sobre cómo calcularlo.
    total_anticipos = "N/A" # Placeholder

    # Fetch exchange rates
    oficial_rate, paralelo_rate = get_dolarapi_rates()

    # Calculate loss per differential for Oficial and Paralelo rates
    # Assuming 'bcv_rate' and 'monitor_rate' from RevenueEntry are the rates used in the original transaction
    # And we want to compare them against the current 'oficial_rate' and 'paralelo_rate' from DolarAPI
    # For now, using a placeholder calculation based on fytd_ansr_amt
    loss_oficial_data = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(
        total_loss=Sum(F('fytd_ansr_amt') * (oficial_rate - F('bcv_rate')), output_field=models.FloatField())
    )['total_loss'] or 0) if oficial_rate else "N/A"

    loss_paralelo_data = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(
        total_loss=Sum(F('fytd_ansr_amt') * (paralelo_rate - F('monitor_rate')), output_field=models.FloatField())
    )['total_loss'] or 0) if paralelo_rate else "N/A"

    # --- Partner Specification Section ---
    partner_spec_data = None
    if selected_partner:
        # These are the entries for the selected partner, filtered by the latest date.
        # Note: revenue_entries_for_kpis is already filtered by the latest date if no week is selected.
        partner_revenue_entries = revenue_entries_for_kpis

        partner_spec_num_engagements = partner_revenue_entries.values('contract').distinct().count()
        partner_spec_num_clients = partner_revenue_entries.values('client').distinct().count()
        partner_spec_client_list = list(partner_revenue_entries.values_list('client__name', flat=True).distinct())

        # Fetch Total Revenue Days P CP from the database for the latest date
        # Get the first entry for the partner to get the unique revenue days value
        first_entry = partner_revenue_entries.first()
        revenue_days_val = first_entry.total_revenue_days_p_cp if first_entry else 0

        partner_spec_data = {
            'num_engagements': partner_spec_num_engagements,
            'num_clients': partner_spec_num_clients,
            'client_list': partner_spec_client_list,
            'revenue_days': f"{revenue_days_val:,.2f}", # Formatting the value
        }

        if selected_client:
            # Entries for the selected client (already filtered by partner)
            client_revenue_entries = partner_revenue_entries.filter(client__name=selected_client)
            
            # Calculate MTD ANSR Sintetico for the selected client
            current_month_start = timezone.now().date().replace(day=1)
            mtd_ansr_sintetico = client_revenue_entries.filter(date__gte=current_month_start).aggregate(
                Sum('fytd_ansr_sintetico')
            )['fytd_ansr_sintetico__sum'] or 0
            
            partner_spec_data['mtd_ansr_sintetico'] = "${:,.2f}".format(mtd_ansr_sintetico)

    context = {
        'ansr_sintetico': ansr_sintetico,
        'total_clients': total_clients,
        'total_engagements': total_engagements,
        'fytd_revenue': fytd_revenue,
        'mtd_revenue': mtd_revenue,
        'daily_revenue': daily_revenue,
        'total_collections': total_collections,
        'total_billing': total_billing,
        'active_employees_venezuela': active_employees_venezuela,
        'top_partners': top_partners,
        'loss_per_differential': loss_per_differential,
        'area_labels': json.dumps(area_labels),
        'area_data': json.dumps(area_data),
        'client_labels': json.dumps(client_labels),
        'client_data': json.dumps(client_data),
        'trend_labels': json.dumps(trend_labels),
        'trend_data': json.dumps(trend_data),
        'recent_entries': recent_entries,

        'partners': partners,
        'managers': managers,
        'areas': areas,
        'sub_areas': sub_areas,
        'clients': clients,
        'available_weeks': available_weeks,

        'selected_partner': selected_partner,
        'selected_manager': selected_manager,
        'selected_area': selected_area,
        'selected_sub_area': selected_sub_area,
        'selected_client': selected_client,
        'selected_week': selected_week_filter, # Pass the selected_week_filter to the template

        'highlights': highlights,
        'top_clients_table': top_clients_table,

        # Nuevas métricas para el contexto
        'partner_distribution_labels': json.dumps(partner_distribution_labels),
        'partner_distribution_data': json.dumps(partner_distribution_data),
        'cartera_moneda_extranjera': cartera_moneda_extranjera,
        'cartera_local_ajustada': cartera_local_ajustada,
        'total_cxc': total_cxc,
        'promedio_antiguedad': promedio_antiguedad,
        'unbilled_labels': json.dumps(unbilled_labels),
        'unbilled_data': json.dumps(unbilled_data),
        'total_anticipos': total_anticipos,

        # Macro Section Data
        'macro_total_clients': "{:,.0f}".format(macro_total_clients),
        'macro_total_ansr_sintetico': "${:,.2f}".format(macro_total_ansr_sintetico),
        'macro_margin': "${:,.2f}".format(macro_margin),
        'macro_margin_percentage': "{:.2f}%".format(macro_margin_percentage),
        'macro_rph': "${:,.2f}".format(macro_rph),
        'macro_monthly_tracker': "${:,.2f}".format(macro_monthly_tracker),

        # Exchange rates
        'oficial_rate': oficial_rate,
        'paralelo_rate': paralelo_rate,

        # Loss per differential breakdown
        'loss_oficial_data': loss_oficial_data,
        'loss_paralelo_data': loss_paralelo_data,
        'partner_spec_data': partner_spec_data,
    }
    return render(request, 'core_dashboard/dashboard.html', context)
