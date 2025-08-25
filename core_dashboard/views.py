

# # Create users if they don't exist
# if not User.objects.filter(username='admin').exists():
#     User.objects.create_user('admin', password='admin')
# if not User.objects.filter(username='dev').exists():
#     User.objects.create_user('dev', password='dev')

from django.shortcuts import render, redirect
from django.db.models import Sum, Count, Q, F, ExpressionWrapper, DecimalField, FloatField
from django.db.models.functions import TruncWeek, Coalesce
from django.utils import timezone
import datetime
import json
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ey_analytics_engine import fetch_all_data, generate_dashboard_analytics
import plotly.graph_objects as go
from django.conf import settings
import os
import traceback
import subprocess

from .data_processor import process_uploaded_files
from .models import UploadHistory, RevenueEntry, Client, Area, SubArea, Contract, ExchangeRate
from .utils import get_fiscal_month_year


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
            ALLOWED_EXTENSIONS = {'.csv', '.xls', '.xlsx', '.xlsb'}
            for f in [engagement_file, dif_file, revenue_days_file]:
                ext = os.path.splitext(f.name)[1]
                if ext.lower() not in ALLOWED_EXTENSIONS:
                    raise ValueError(f"File type {ext} is not allowed.")

            # Create history directory for the given date
            history_dir = os.path.join(settings.MEDIA_ROOT, 'historico_de_final_database', upload_date.strftime('%Y-%m-%d'))
            os.makedirs(history_dir, exist_ok=True)

            from django.core.files.storage import FileSystemStorage
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
                uploaded_by=None
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


def tables_view(request):
    return render(request, 'core_dashboard/tables.html')


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


def messaging_view(request):
    # Get distinct partners, managers, and areas for messaging
    partners = RevenueEntry.objects.values_list('engagement_partner', flat=True).distinct().exclude(engagement_partner__isnull=True).exclude(Q(engagement_partner__exact='')).order_by('engagement_partner')
    managers = RevenueEntry.objects.values_list('engagement_manager', flat=True).distinct().exclude(engagement_manager__isnull=True).exclude(Q(engagement_manager__exact='')).order_by('engagement_manager')
    areas = Area.objects.values_list('name', flat=True).distinct().exclude(name__isnull=True).exclude(Q(name__exact='')).order_by('name')
    context = {
        'partners': partners,
        'managers': managers,
        'areas': areas
    }
    return render(request, 'core_dashboard/messaging.html', context)


def dashboard_view(request):
    # Get filter parameters from request
    selected_partner = request.GET.get('partner')
    selected_manager = request.GET.get('manager')
    selected_area = request.GET.get('service_line')
    selected_sub_area = request.GET.get('sub_service_line')
    selected_client = request.GET.get('client')
    selected_week_filter = request.GET.get('week') # Renamed to avoid conflict

    print(f"DEBUG: selected_partner: {selected_partner}")
    print(f"DEBUG: selected_manager: {selected_manager}")
    print(f"DEBUG: selected_area: {selected_area}")
    print(f"DEBUG: selected_sub_area: {selected_sub_area}")
    print(f"DEBUG: selected_client: {selected_client}")
    print(f"DEBUG: selected_week_filter: {selected_week_filter}")
    print(f"DEBUG: selected_partner: {selected_partner}")
    print(f"DEBUG: selected_manager: {selected_manager}")
    print(f"DEBUG: selected_area: {selected_area}")
    print(f"DEBUG: selected_sub_area: {selected_sub_area}")
    print(f"DEBUG: selected_client: {selected_client}")
    print(f"DEBUG: selected_week_filter: {selected_week_filter}")

    # Queryset for historical trend (uses all data)
    all_revenue_entries = RevenueEntry.objects.all()

    # Base queryset for KPIs. Start with all entries.
    base_revenue_entries = RevenueEntry.objects.all()

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

    # Date filtering logic for the entire page
    if selected_week_filter:
        friday_date = datetime.datetime.strptime(selected_week_filter, '%Y-%m-%d').date()
        start_of_week = friday_date - datetime.timedelta(days=friday_date.weekday())
        end_of_week = start_of_week + datetime.timedelta(days=6)
        base_revenue_entries = base_revenue_entries.filter(date__range=[start_of_week, end_of_week])
    elif available_weeks:
        # Default to the most recent week if no week is selected
        most_recent_week = available_weeks[-1]
        friday_date = datetime.datetime.strptime(most_recent_week, '%Y-%m-%d').date()
        print(f"DEBUG: base_revenue_entries count after week filter: {base_revenue_entries.count()}")
        start_of_week = friday_date - datetime.timedelta(days=friday_date.weekday())
        end_of_week = start_of_week + datetime.timedelta(days=6)
        base_revenue_entries = base_revenue_entries.filter(date__range=[start_of_week, end_of_week])
    else:
        base_revenue_entries = RevenueEntry.objects.none()
    print(f"DEBUG: base_revenue_entries count after week filter: {base_revenue_entries.count()}")

    # --- Macro Section Calculations (always on the full dataset for the selected date) ---
    macro_revenue_entries = base_revenue_entries
    if selected_area:
        macro_revenue_entries = macro_revenue_entries.filter(area__name=selected_area)
    if selected_sub_area:
        macro_revenue_entries = macro_revenue_entries.filter(sub_area__name=selected_sub_area)
    if selected_client:
        macro_revenue_entries = macro_revenue_entries.filter(client__name=selected_client)
    
    macro_total_clients = macro_revenue_entries.values('client').distinct().count()
    macro_ansr_fytd = macro_revenue_entries.aggregate(Sum('fytd_ansr_amt'))['fytd_ansr_amt__sum'] or 0
    macro_total_direct_cost = macro_revenue_entries.aggregate(Sum('fytd_direct_cost_amt'))['fytd_direct_cost_amt__sum'] or 0
    macro_margin = macro_ansr_fytd - macro_total_direct_cost
    macro_margin_percentage = (macro_margin / macro_ansr_fytd * 100) if macro_ansr_fytd else 0
    macro_total_charged_hours = macro_revenue_entries.aggregate(Sum('fytd_charged_hours'))['fytd_charged_hours__sum'] or 0
    macro_rph = (macro_ansr_fytd / macro_total_charged_hours) if macro_total_charged_hours else 0
    macro_mtd_charged_hours = macro_revenue_entries.aggregate(Sum('mtd_charged_hours'))['mtd_charged_hours__sum'] or 0
    macro_monthly_tracker = macro_mtd_charged_hours * macro_rph
    total_mtd_charged_hours = macro_revenue_entries.aggregate(Sum('mtd_charged_hours'))['mtd_charged_hours__sum'] or 0
    total_fytd_charged_hours = macro_revenue_entries.aggregate(Sum('fytd_charged_hours'))['fytd_charged_hours__sum'] or 0
    macro_diferencial_final = macro_revenue_entries.aggregate(Sum('fytd_diferencial_final'))['fytd_diferencial_final__sum'] or 0

    # Sum of 'fytd_diferencial_final' per partner
    diferencial_final_by_partner = base_revenue_entries.values('engagement_partner').annotate(
        total_diferencial_final=Coalesce(Sum('fytd_diferencial_final'), 0.0)
    ).order_by('-total_diferencial_final')

    # --- Filtered Section (for partner/manager specific views) ---
    revenue_entries_for_kpis = base_revenue_entries
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
    print(f"DEBUG: revenue_entries_for_kpis count after all filters: {revenue_entries_for_kpis.count()}")
    print(f"DEBUG: revenue_entries_for_kpis first entry: {revenue_entries_for_kpis.first()}")

    # KPIs for the filtered view
    ansr_sintetico = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(Sum('revenue'))['revenue__sum'] or 0)
    total_clients = "{:,.0f}".format(revenue_entries_for_kpis.values('client').distinct().count())
    total_engagements = "{:,.0f}".format(revenue_entries_for_kpis.values('contract').distinct().count())

    # Charged hours by partner and manager (based on the filtered data)
    fytd_charged_hours_by_partner = revenue_entries_for_kpis.values('engagement_partner').annotate(
        total_fytd_charged_hours=Sum('fytd_charged_hours')
    ).order_by('-total_fytd_charged_hours')

    mtd_charged_hours_by_partner = revenue_entries_for_kpis.values('engagement_partner').annotate(
        total_mtd_charged_hours=Sum('mtd_charged_hours')
    ).order_by('-total_mtd_charged_hours')

    fytd_charged_hours_by_manager = revenue_entries_for_kpis.values('engagement_manager').annotate(
        total_fytd_charged_hours=Sum('fytd_charged_hours')
    ).order_by('-total_fytd_charged_hours')

    mtd_charged_hours_by_manager = revenue_entries_for_kpis.values('engagement_manager').annotate(
        total_mtd_charged_hours=Sum('mtd_charged_hours')
    ).order_by('-total_mtd_charged_hours')

    # Add FYTD_ChargedHours and MTD_Charged to the partner filter
    # This will be used in the template to display the data
    fytd_charged_hours_by_partner_with_labels = []
    mtd_charged_hours_by_partner_with_labels = []

    for partner in fytd_charged_hours_by_partner:
        fytd_charged_hours_by_partner_with_labels.append({
            'engagement_partner': partner['engagement_partner'],
            'total_fytd_charged_hours': partner['total_fytd_charged_hours'],
            'label': 'Horas Cargadas (FYTD)'
        })

    for partner in mtd_charged_hours_by_partner:
        mtd_charged_hours_by_partner_with_labels.append({
            'engagement_partner': partner['engagement_partner'],
            'total_mtd_charged_hours': partner['total_mtd_charged_hours'],
            'label': 'Horas Cargadas (MTD)'
        })

    # --- All Partners Ranked (for flip card) ---
    all_partners_ranked = base_revenue_entries.values('engagement_partner').annotate(
        total_revenue=Sum('revenue')
    ).order_by('-total_revenue').exclude(engagement_partner__isnull=True).exclude(engagement_partner__exact='')
    for p in all_partners_ranked:
        p['total_revenue'] = "${:,.2f}".format(p['total_revenue'] or 0)

    # Get all FYTD charged hours by partner to merge with the ranked list
    all_fytd_charged_hours_by_partner = base_revenue_entries.values('engagement_partner').annotate(
        total_fytd_charged_hours=Sum('fytd_charged_hours')
    ).order_by('engagement_partner')

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
        p['total_revenue'] = "${:,.2f}".format(p['total_revenue'] or 0)

    # Top 5 Clients by Revenue (for table display)
    top_clients_table = revenue_entries_for_kpis.values('client__name').annotate(total_revenue=Sum('revenue')).order_by('-total_revenue')[:5]
    # Format revenue for top_clients_table
    for c in top_clients_table:
        c['total_revenue'] = "${:,.2f}".format(c['total_revenue'] or 0)

    # Calculate "Loss per differential"
    # Assuming 'bcv_rate' and 'monitor_rate' are fields in RevenueEntry
    # Loss = (BCV Rate - Monitor Rate) * Revenue
    loss_per_differential = "${:,.2f}".format(revenue_entries_for_kpis.annotate(
        differential_loss=ExpressionWrapper((F('bcv_rate') - F('monitor_rate')) * F('revenue'), output_field=DecimalField())
    ).aggregate(Sum('differential_loss'))['differential_loss__sum'] or 0)

    # Revenue by Area
    revenue_by_area = revenue_entries_for_kpis.values('area__name').annotate(total_revenue=Sum('revenue')).order_by('-total_revenue')
    area_labels = [item['area__name'] for item in revenue_by_area]
    area_data = [float(item['total_revenue'] or 0) for item in revenue_by_area]

    # Top 5 Clients by Revenue (for chart)
    top_clients_chart = revenue_entries_for_kpis.values('client__name').annotate(total_revenue=Sum('revenue')).order_by('-total_revenue')[:5]
    client_labels = [item['client__name'] for item in top_clients_chart]
    client_data = [float(item['total_revenue'] or 0) for item in top_clients_chart]

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
        entry.revenue_formatted = "${:,.2f}".format(entry.revenue or 0)

    # Get distinct values for filters, excluding None and empty strings
    partners = RevenueEntry.objects.values_list('engagement_partner', flat=True).distinct().exclude(engagement_partner__isnull=True).exclude(Q(engagement_partner__exact='')).order_by('engagement_partner')
    managers = RevenueEntry.objects.values_list('engagement_manager', flat=True).distinct().exclude(engagement_manager__isnull=True).exclude(Q(engagement_manager__exact='')).order_by('engagement_manager')
    areas = Area.objects.values_list('name', flat=True).distinct().exclude(name__isnull=True).exclude(Q(name__exact='')).order_by('name')
    sub_areas = SubArea.objects.values_list('name', flat=True).distinct().exclude(name__isnull=True).exclude(Q(name__exact='')).order_by('name')
    clients = Client.objects.values_list('name', flat=True).distinct().exclude(name__isnull=True).exclude(Q(name__exact='')).order_by('name')

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
    macro_total_ansr_sintetico = revenue_entries_for_kpis.aggregate(Sum('fytd_ansr_amt'))['fytd_ansr_amt__sum'] or 0
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

    # 3. Cartera local ajustada (usando fytd_ansr_amt)
    cartera_local_ajustada = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(Sum('fytd_ansr_amt'))['fytd_ansr_amt__sum'] or 0)

    # 4. Total CXC (asumiendo que es la suma de billing - collections, o solo billing si no hay un campo de CXC explícito)
    # Si tienes un campo de CXC directo, por favor, indícalo.
    total_cxc = "${:,.2f}".format(revenue_entries_for_kpis.aggregate(Sum('billing'))['billing__sum'] or 0) # Usando billing como proxy

    # 5. Promedio de antigüedad (requiere un campo de fecha de inicio de contrato/cliente y fecha actual)
    # Por ahora, es un placeholder. Necesito más información sobre cómo calcularlo.
    promedio_antiguedad = "N/A" # Placeholder

    # 6. Unbilled Inventory por Service Line
    # Asumiendo que Unbilled Inventory = FYTD_ANSRAmt - Collections
    unbilled_inventory_by_service_line = revenue_entries_for_kpis.values('engagement_service_line').annotate(
        unbilled_amount=Sum(ExpressionWrapper(F('fytd_ansr_amt') - F('collections'), output_field=FloatField()))
    ).order_by('-unbilled_amount').exclude(engagement_service_line__isnull=True).exclude(engagement_service_line__exact='')

    unbilled_labels = [item['engagement_service_line'] for item in unbilled_inventory_by_service_line]
    unbilled_data = [float(item['unbilled_amount'] or 0) for item in unbilled_inventory_by_service_line]

    # 7. Anticipos (requiere un campo específico para anticipos)
    # Por ahora, es un placeholder. Necesito más información sobre cómo calcularlo.
    total_anticipos = "N/A" # Placeholder

    # --- Partner Specification Section ---
    partner_spec_data = None
    if selected_partner:
        print(f"DEBUG: Entering partner_spec_data block for partner: {selected_partner}")
        partner_revenue_entries = revenue_entries_for_kpis
        print(f"DEBUG: partner_revenue_entries count: {partner_revenue_entries.count()}")
        print(f"DEBUG: partner_revenue_entries first entry: {partner_revenue_entries.first()}")

        partner_spec_num_engagements = partner_revenue_entries.values('contract').distinct().count()
        partner_spec_num_clients = partner_revenue_entries.values('client').distinct().count()

        # Get client list with their revenue for the selected partner
        client_list_with_revenue = partner_revenue_entries.values('client__name').annotate(
            total_revenue=Sum('revenue')
        ).order_by('-total_revenue')

        # Format revenue for the client list
        formatted_client_list = [
            {
                'name': item['client__name'],
                'revenue': "${:,.2f}".format(item['total_revenue'] or 0)
            }
            for item in client_list_with_revenue
        ]

        first_entry = partner_revenue_entries.first()
        revenue_days_val = first_entry.total_revenue_days_p_cp if first_entry else 0

        # Partner-specific ANSR FYTD and MTD
        partner_fytd_ansr_value = partner_revenue_entries.aggregate(Sum('fytd_ansr_amt'))['fytd_ansr_amt__sum'] or 0
        partner_mtd_ansr_value = partner_revenue_entries.aggregate(Sum('mtd_ansr_amt'))['mtd_ansr_amt__sum'] or 0

        # Goals for partner-specific ANSR
        partner_fytd_ansr_goal = 0
        partner_mtd_ansr_goal = 0
        partner_fytd_ansr_completion_percentage = 0
        partner_mtd_ansr_completion_percentage = 0

        try:
            metas_pped_path = os.path.join(settings.BASE_DIR, 'metas_PPED.csv')
            if os.path.exists(metas_pped_path):
                metas_pped_df = pd.read_csv(metas_pped_path)
                # Normalize partner names in DataFrame
                metas_pped_df['Partner'] = metas_pped_df['Partner'].str.strip().str.lower()

                # Normalize selected_partner for comparison
                normalized_selected_partner = selected_partner.strip().lower() if selected_partner else ''

                # Debug: Print available partners in CSV
                print(f"DEBUG: Available partners in CSV: {metas_pped_df['Partner'].unique()}")
                print(f"DEBUG: Looking for partner: {normalized_selected_partner}")

                # Get yearly goal for the selected partner
                partner_yearly_goal_rows = metas_pped_df[
                    (metas_pped_df['Partner'].str.contains(normalized_selected_partner, case=False, na=False)) &
                    (metas_pped_df['Mes'] == 'Total')
                ]
                if not partner_yearly_goal_rows.empty:
                    partner_fytd_ansr_goal = partner_yearly_goal_rows['ANSR Goal PPED'].sum()
                    partner_fytd_hours_goal = partner_yearly_goal_rows['Horas Goal PPED'].sum()
                    print(f"DEBUG: Found FYTD goal for {normalized_selected_partner}: {partner_fytd_ansr_goal}")
                    print(f"DEBUG: Found FYTD hours goal for {normalized_selected_partner}: {partner_fytd_hours_goal}")
                else:
                    partner_fytd_hours_goal = 0
                    print(f"DEBUG: No FYTD goal found for {normalized_selected_partner}")

                # Get monthly goal for the selected partner based on fiscal month
                current_report_date = None
                if selected_week_filter:
                    current_report_date = datetime.datetime.strptime(selected_week_filter, '%Y-%m-%d').date()
                elif available_weeks:
                    current_report_date = datetime.datetime.strptime(available_weeks[-1], '%Y-%m-%d').date()

                if current_report_date:
                    fiscal_month_name_for_goal = get_fiscal_month_year(current_report_date)
                    partner_monthly_goal_rows = metas_pped_df[
                        (metas_pped_df['Partner'].str.contains(normalized_selected_partner, case=False, na=False)) &
                        (metas_pped_df['Mes'] == fiscal_month_name_for_goal)
                    ]
                    if not partner_monthly_goal_rows.empty:
                        partner_mtd_ansr_goal = partner_monthly_goal_rows['ANSR Goal PPED'].sum()
                        partner_mtd_hours_goal = partner_monthly_goal_rows['Horas Goal PPED'].sum()
                        print(f"DEBUG: Found MTD goal for {normalized_selected_partner} in {fiscal_month_name_for_goal}: {partner_mtd_ansr_goal}")
                        print(f"DEBUG: Found MTD hours goal for {normalized_selected_partner} in {fiscal_month_name_for_goal}: {partner_mtd_hours_goal}")
                    else:
                        partner_mtd_hours_goal = 0
                        print(f"DEBUG: No MTD goal found for {normalized_selected_partner} in {fiscal_month_name_for_goal}")

            # Calculate completion percentages for partner
            if partner_fytd_ansr_goal > 0:
                partner_fytd_ansr_completion_percentage = (partner_fytd_ansr_value / partner_fytd_ansr_goal) * 100
            if partner_mtd_ansr_goal > 0:
                partner_mtd_ansr_completion_percentage = (partner_mtd_ansr_value / partner_mtd_ansr_goal) * 100

            # Calculate charged hours completion percentages
            partner_fytd_hours_completion_percentage = 0
            partner_mtd_hours_completion_percentage = 0

            # Get partner's actual charged hours
            partner_fytd_charged_hours = partner_revenue_entries.aggregate(Sum('fytd_charged_hours'))['fytd_charged_hours__sum'] or 0
            partner_mtd_charged_hours = partner_revenue_entries.aggregate(Sum('mtd_charged_hours'))['mtd_charged_hours__sum'] or 0

            if partner_fytd_hours_goal > 0:
                partner_fytd_hours_completion_percentage = (partner_fytd_charged_hours / partner_fytd_hours_goal) * 100
            if partner_mtd_hours_goal > 0:
                partner_mtd_hours_completion_percentage = (partner_mtd_charged_hours / partner_mtd_hours_goal) * 100

            # Add goals per partner using the same format as ANSR information but with metas_PPED data
            partner_spec_data['partner_fytd_hours_goal'] = partner_fytd_hours_goal
            partner_spec_data['partner_mtd_hours_goal'] = partner_mtd_hours_goal
            partner_spec_data['partner_fytd_hours_completion_percentage'] = partner_fytd_hours_completion_percentage
            partner_spec_data['partner_mtd_hours_completion_percentage'] = partner_mtd_hours_completion_percentage

        except Exception as e:
            print(f"Error processing partner goals: {e}")
            traceback.print_exc()

        partner_spec_data = {
            'num_engagements': partner_spec_num_engagements,
            'num_clients': partner_spec_num_clients,
            'client_list': formatted_client_list, # Use the new formatted list
            'revenue_days': f"{revenue_days_val:,.2f}",
            'partner_fytd_ansr_value': partner_fytd_ansr_value,
            'partner_fytd_ansr_goal': partner_fytd_ansr_goal,
            'partner_fytd_ansr_completion_percentage': partner_fytd_ansr_completion_percentage,
            'partner_mtd_ansr_value': partner_mtd_ansr_value,
            'partner_mtd_ansr_goal': partner_mtd_ansr_goal,
            'partner_mtd_ansr_completion_percentage': partner_mtd_ansr_completion_percentage,
            'partner_fytd_charged_hours': partner_fytd_charged_hours,
            'partner_fytd_hours_goal': partner_fytd_hours_goal,
            'partner_fytd_hours_completion_percentage': partner_fytd_hours_completion_percentage,
            'partner_mtd_charged_hours': partner_mtd_charged_hours,
            'partner_mtd_hours_goal': partner_mtd_hours_goal,
            'partner_mtd_hours_completion_percentage': partner_mtd_hours_completion_percentage,
        }

        if selected_client:
            client_revenue_entries = partner_revenue_entries.filter(client__name=selected_client)
            current_month_start = timezone.now().date().replace(day=1)
            mtd_ansr_amt = client_revenue_entries.filter(date__gte=current_month_start).aggregate(
                Sum('mtd_ansr_amt')
            )['mtd_ansr_amt__sum'] or 0
            partner_spec_data['mtd_ansr_amt'] = "${:,.2f}".format(mtd_ansr_amt)

        # --- New Perdida Diferencial Calculations for Partner View ---
        # 1. Total Perdida Diferencial for the partner
        partner_perdida_diferencial = partner_revenue_entries.aggregate(
            total_perdida=Sum('fytd_diferencial_final')
        )['total_perdida'] or 0
        partner_spec_data['total_perdida_diferencial'] = "${:,.2f}".format(partner_perdida_diferencial)

        # 2. Perdida Diferencial per Client for the partner
        perdida_per_client = partner_revenue_entries.values('client__name').annotate(
            perdida=Sum('fytd_diferencial_final')
        ).order_by('-perdida')
        partner_spec_data['perdida_per_client'] = [
            {'client_name': item['client__name'], 'perdida': "${:,.2f}".format(item['perdida'] or 0)}
            for item in perdida_per_client
        ]

        # 3. Top 5 Engagements by Perdida Diferencial for the partner
        top_engagements_perdida = partner_revenue_entries.values('contract__name').annotate(
            perdida=Sum('fytd_diferencial_final')
        ).order_by('-perdida')[:5]
        partner_spec_data['top_engagements_perdida'] = [
            {'engagement_name': item['contract__name'], 'perdida': "${:,.2f}".format(item['perdida'] or 0)}
            for item in top_engagements_perdida
        ]

    # Fetch historical exchange rates for charting
    exchange_rates_data = ExchangeRate.objects.all().order_by('date')
    exchange_dates = [er.date.strftime('%Y-%m-%d') for er in exchange_rates_data]
    oficial_rates_history = [float(er.oficial_rate) if er.oficial_rate else None for er in exchange_rates_data]
    paralelo_rates_history = [float(er.paralelo_rate) if er.paralelo_rate else None for er in exchange_rates_data]
    differential_history = [float(er.differential) if er.differential else None for er in exchange_rates_data]

    context = {
        'total_fytd_charged_hours': "{:,.0f}".format(total_fytd_charged_hours),
        'total_mtd_charged_hours': "{:,.0f}".format(total_mtd_charged_hours),
        'fytd_charged_hours_by_partner': fytd_charged_hours_by_partner,
        'mtd_charged_hours_by_partner': mtd_charged_hours_by_partner,
        'fytd_charged_hours_by_manager': fytd_charged_hours_by_manager,
        'mtd_charged_hours_by_manager': mtd_charged_hours_by_manager,
        'fytd_charged_hours_by_partner_with_labels': fytd_charged_hours_by_partner_with_labels,
        'mtd_charged_hours_by_partner_with_labels': mtd_charged_hours_by_partner_with_labels,
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
        'macro_total_ansr_sintetico': macro_total_ansr_sintetico,
        'macro_margin': "${:,.2f}".format(macro_margin),
        'macro_margin_percentage': "{:.2f}%".format(macro_margin_percentage),
        'macro_rph': "${:,.2f}".format(macro_rph),
        'macro_monthly_tracker': macro_monthly_tracker,
        'total_fytd_charged_hours': "{:,.0f}".format(total_fytd_charged_hours),
        'total_mtd_charged_hours': "{:,.0f}".format(total_mtd_charged_hours),
        'macro_diferencial_final': macro_diferencial_final,

        # New context for flip card
        'all_partners_ranked': all_partners_ranked,
        'all_fytd_charged_hours_by_partner': all_fytd_charged_hours_by_partner,

        # New: Accumulated sums for MTD_ChargedHours and FYTD_ChargedHours
        'total_mtd_charged_hours': "{:,.0f}".format(total_mtd_charged_hours),
        'total_fytd_charged_hours': "{:,.0f}".format(total_fytd_charged_hours),

        # New: Accumulated sums for MTD_ChargedHours and FYTD_ChargedHours
        'total_mtd_charged_hours': "{:,.0f}".format(total_mtd_charged_hours),
        'total_fytd_charged_hours': "{:,.0f}".format(total_fytd_charged_hours),

        'partner_spec_data': partner_spec_data,

        # Exchange Rate History for Chart
        'exchange_dates': json.dumps(exchange_dates),
        'oficial_rates_history': json.dumps(oficial_rates_history),
        'paralelo_rates_history': json.dumps(paralelo_rates_history),
        'differential_history': json.dumps(differential_history),
        'diferencial_final_by_partner': diferencial_final_by_partner,
    }
    # --- Goals Calculation ---
    ansr_fytd_goal = 0
    ansr_mtd_goal = 0
    ansr_fytd_completion_percentage = 0
    ansr_mtd_completion_percentage = 0

    try:
        metas_sl_path = os.path.join(settings.BASE_DIR, 'metas_SL.csv')
        if os.path.exists(metas_sl_path):
            metas_sl_df = pd.read_csv(metas_sl_path)

            # Determine the current report date
            current_report_date = None
            if selected_week_filter:
                current_report_date = datetime.datetime.strptime(selected_week_filter, '%Y-%m-%d').date()
            elif available_weeks:
                current_report_date = datetime.datetime.strptime(available_weeks[-1], '%Y-%m-%d').date()

            if current_report_date:
                # Get fiscal month and year using the new utility function
                fiscal_month_name_for_goal = get_fiscal_month_year(current_report_date)

                # Get yearly goal (ANSR FYTD Goal)
                yearly_goal_row = metas_sl_df[(metas_sl_df['SL'] == 'Total general') & (metas_sl_df['Mes'] == 'Total')]
                if not yearly_goal_row.empty:
                    ansr_fytd_goal = yearly_goal_row['ANSR Goal'].iloc[0]
                    hours_fytd_goal = yearly_goal_row['Horas Goal'].iloc[0]

                # Get monthly goal (ANSR MTD Goal) based on fiscal month
                monthly_goal_row = metas_sl_df[(metas_sl_df['SL'] == 'Total general') & (metas_sl_df['Mes'] == fiscal_month_name_for_goal)]
                if not monthly_goal_row.empty:
                    ansr_mtd_goal = monthly_goal_row['ANSR Goal'].iloc[0]
                    hours_mtd_goal = monthly_goal_row['Horas Goal'].iloc[0]

            # Calculate completion percentages
            # ANSR FYTD (formerly ansr_sintetico)
            if ansr_fytd_goal > 0:
                ansr_fytd_completion_percentage = (macro_total_ansr_sintetico / ansr_fytd_goal) * 100

            # ANSR MTD (formerly monthly_tracker)
            if ansr_mtd_goal > 0:
                ansr_mtd_completion_percentage = (macro_monthly_tracker / ansr_mtd_goal) * 100

            # Hours FYTD
            hours_fytd_completion_percentage = 0
            if 'hours_fytd_goal' in locals() and hours_fytd_goal > 0:
                total_fytd_charged_hours_float = float(str(total_fytd_charged_hours).replace(',', ''))
                hours_fytd_completion_percentage = (total_fytd_charged_hours_float / hours_fytd_goal) * 100

            # Hours MTD
            hours_mtd_completion_percentage = 0
            if 'hours_mtd_goal' in locals() and hours_mtd_goal > 0:
                total_mtd_charged_hours_float = float(str(total_mtd_charged_hours).replace(',', ''))
                hours_mtd_completion_percentage = (total_mtd_charged_hours_float / hours_mtd_goal) * 100
            else:
                hours_mtd_completion_percentage = 0

    except Exception as e:
        print(f"Error processing goals: {e}")
        traceback.print_exc() # Print full traceback for debugging

    context['ansr_fytd_value'] = macro_total_ansr_sintetico
    context['ansr_fytd_goal'] = ansr_fytd_goal
    context['ansr_fytd_completion_percentage'] = ansr_fytd_completion_percentage
    context['ansr_mtd_value'] = macro_monthly_tracker
    context['ansr_mtd_goal'] = ansr_mtd_goal
    context['ansr_mtd_completion_percentage'] = ansr_mtd_completion_percentage

    # Add hours goals to context
    context['hours_fytd_goal'] = hours_fytd_goal
    context['hours_mtd_goal'] = hours_mtd_goal
    context['hours_fytd_completion_percentage'] = hours_fytd_completion_percentage
    context['hours_mtd_completion_percentage'] = hours_mtd_completion_percentage
    
    print(f"DEBUG: Final context keys: {context.keys()}")
    if 'partner_spec_data' in context and context['partner_spec_data']:
        print(f"DEBUG: partner_spec_data is present in context.")
        for key, value in context['partner_spec_data'].items():
            print(f"DEBUG:   partner_spec_data['{key}']: {value}")
    else:
        print(f"DEBUG: partner_spec_data is NOT present or is empty in context.")

    return render(request, 'core_dashboard/dashboard.html', context)


def data_downloads_view(request):
    DATA_DIR = os.path.join(settings.MEDIA_ROOT, 'historico_de_final_database')
    
    final_databases = []
    report_dates = []
    if os.path.exists(DATA_DIR):
        weekly_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        report_dates = sorted(weekly_dirs, reverse=True)
        for d in weekly_dirs:
            dir_path = os.path.join(DATA_DIR, d)
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and 'Final_Database' in f and f.endswith('.csv')]
            for f in files:
                final_databases.append({'date': d, 'filename': f, 'path': f'{d}/{f}'})

    # Handle file download
    download_path = request.GET.get('download')
    download_format = request.GET.get('format', 'csv')
    if download_path:
        file_path = os.path.join(DATA_DIR, download_path)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if download_format == 'excel':
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                output.seek(0)
                response = HttpResponse(output.read(), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                response['Content-Disposition'] = f'attachment; filename={os.path.splitext(os.path.basename(file_path))[0]}.xlsx'
                return response
            else: # CSV
                from django.http import HttpResponse
                response = HttpResponse(df.to_csv(index=False), content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename={os.path.basename(file_path)}'
                return response

    # Handle file preview
    selected_date = request.GET.get('report_date')
    df_html = None
    preview_path = None
    if selected_date:
        # Find the final database for the selected date
        for db in final_databases:
            if db['date'] == selected_date:
                preview_path = db['path']
                file_path = os.path.join(DATA_DIR, preview_path)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df_html = df.to_html(classes='table table-dark table-striped table-hover', index=False)
                break

    context = {
        'final_databases': sorted(final_databases, key=lambda x: x['date'], reverse=True),
        'report_dates': report_dates,
        'df_html': df_html,
        'selected_date': selected_date,
        'preview_path': preview_path,
    }
    return render(request, 'core_dashboard/data_downloads.html', context)


def help_view(request):
    return render(request, 'core_dashboard/help.html')


def settings_view(request):
    return render(request, 'core_dashboard/settings.html')
''