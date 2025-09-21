from flask import Flask, request, jsonify, render_template
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime, timedelta
import os
import numpy as np
from sklearn.cluster import KMeans
import math

# Import ML forecasting module
try:
    from demand_forecasting_model import MLInventoryManager, ForecastResult, TransferSuggestion
    ML_ENABLED = True
except ImportError as e:
    print(f"Warning: ML module not available: {e}")
    ML_ENABLED = False

app = Flask(__name__)

# Global variables to store data and ML manager
inventory_data = None
ml_manager = None

# Initialize ML manager if available
if ML_ENABLED:
    try:
        ml_manager = MLInventoryManager()
        print("✅ ML Inventory Manager initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize ML manager: {e}")
        ML_ENABLED = False

def process_inventory_data(df):
    """Process the inventory data and calculate various metrics"""
    # Convert date columns
    df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
    df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
    
    # Calculate days until expiry
    today = pd.Timestamp.now().normalize()
    df['DaysUntilExpiry'] = (df['ExpiryDate'] - today).dt.days
    
    # Categorize stock status
    def categorize_stock(row):
        if row['Quantity'] == 0:
            return 'Out of Stock'
        elif row['DaysUntilExpiry'] < 0:
            return 'Expired'
        elif row['DaysUntilExpiry'] <= 30:
            return 'Expiring Soon'
        elif row['Quantity'] <= 10:
            return 'Low Stock'
        else:
            return 'In Stock'
    
    df['StockStatus'] = df.apply(categorize_stock, axis=1)
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transfer')
def transfer():
    return render_template('transfer.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    global inventory_data
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Validate required columns
        required_columns = ['Store', 'Medicine', 'Quantity', 'ExpiryDate', 'LastUpdated']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': f'Missing required columns. Expected: {required_columns}'}), 400
        
        # Process the data
        inventory_data = process_inventory_data(df)
        
        return jsonify({
            'message': '✅ CSV uploaded successfully to PillPilot',
            'rows': len(inventory_data),
            'stores': inventory_data['Store'].nunique(),
            'medicines': inventory_data['Medicine'].nunique()
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({'error': f'Error processing CSV: {str(e)}', 'details': error_details}), 400

@app.route('/api/inventory-summary')
def inventory_summary():
    if inventory_data is None:
        return jsonify({'error': 'No data available. Please upload a CSV file first.'}), 400
    
    # Store-wise summary
    store_summary = inventory_data.groupby('Store').agg({
        'Quantity': 'sum',
        'Medicine': 'nunique',
        'StockStatus': lambda x: (x == 'Out of Stock').sum()
    }).rename(columns={'Medicine': 'UniqueMedicines', 'StockStatus': 'OutOfStockCount'})
    
    # Medicine-wise summary
    medicine_summary = inventory_data.groupby('Medicine').agg({
        'Quantity': 'sum',
        'Store': 'nunique',
        'StockStatus': lambda x: (x == 'Out of Stock').sum()
    }).rename(columns={'Store': 'StoresWithStock', 'StockStatus': 'OutOfStockCount'})
    
    # Overall statistics
    total_quantity = inventory_data['Quantity'].sum()
    expired_count = (inventory_data['StockStatus'] == 'Expired').sum()
    out_of_stock_count = (inventory_data['StockStatus'] == 'Out of Stock').sum()
    expiring_soon_count = (inventory_data['StockStatus'] == 'Expiring Soon').sum()
    
    return jsonify({
        'store_summary': store_summary.to_dict('index'),
        'medicine_summary': medicine_summary.to_dict('index'),
        'overall_stats': {
            'total_quantity': int(total_quantity),
            'expired_count': int(expired_count),
            'out_of_stock_count': int(out_of_stock_count),
            'expiring_soon_count': int(expiring_soon_count)
        }
    })

@app.route('/api/expired-medicines')
def expired_medicines():
    if inventory_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    expired = inventory_data[inventory_data['StockStatus'] == 'Expired']
    return jsonify(expired.to_dict('records'))

@app.route('/api/stock-levels')
def stock_levels():
    if inventory_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    # Group by store and medicine for detailed view
    stock_levels = inventory_data.groupby(['Store', 'Medicine']).agg({
        'Quantity': 'first',
        'StockStatus': 'first',
        'DaysUntilExpiry': 'first',
        'ExpiryDate': 'first'
    }).reset_index()
    
    return jsonify(stock_levels.to_dict('records'))

@app.route('/api/charts/stock-by-store')
def stock_by_store_chart():
    if inventory_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    # Create bar chart for stock by store
    store_totals = inventory_data.groupby('Store')['Quantity'].sum().reset_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=store_totals['Store'],
            y=store_totals['Quantity'],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        )
    ])
    
    fig.update_layout(
        title='Total Stock by Store',
        xaxis_title='Store',
        yaxis_title='Total Quantity',
        template='plotly_white'
    )
    
    return jsonify(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

@app.route('/api/charts/medicine-distribution')
def medicine_distribution_chart():
    if inventory_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    # Create pie chart for medicine distribution
    medicine_totals = inventory_data.groupby('Medicine')['Quantity'].sum().reset_index()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=medicine_totals['Medicine'],
            values=medicine_totals['Quantity'],
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title='Medicine Distribution Across All Stores',
        template='plotly_white'
    )
    
    return jsonify(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

@app.route('/api/charts/stock-status')
def stock_status_chart():
    if inventory_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    # Create pie chart for stock status
    status_counts = inventory_data['StockStatus'].value_counts()
    
    colors = {
        'In Stock': '#2ECC71',
        'Low Stock': '#F39C12',
        'Out of Stock': '#E74C3C',
        'Expired': '#8E44AD',
        'Expiring Soon': '#E67E22'
    }
    
    fig = go.Figure(data=[
        go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            marker_colors=[colors.get(status, '#95A5A6') for status in status_counts.index]
        )
    ])
    
    fig.update_layout(
        title='Stock Status Distribution',
        template='plotly_white'
    )
    
    return jsonify(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def calculate_transfer_suggestions():
    """
    Calculate intelligent transfer suggestions based on:
    1. Stock imbalances between stores
    2. Expiry dates (prioritize moving items that expire sooner)
    3. Store capacity and demand patterns
    4. Minimize total transfers while maximizing impact
    """
    if inventory_data is None:
        return []
    
    suggestions = []
    
    # Group by medicine to analyze each medicine separately
    medicines = inventory_data['Medicine'].unique()
    
    for medicine in medicines:
        medicine_data = inventory_data[inventory_data['Medicine'] == medicine].copy()
        
        # Skip if only one store has this medicine
        if len(medicine_data) < 2:
            continue
            
        # Calculate metrics for each store
        medicine_data['Priority'] = 0
        
        # Priority factors:
        # 1. Stock level (higher stock = potential donor)
        # 2. Days until expiry (shorter = higher priority to move)
        # 3. Store needs (out of stock = high priority recipient)
        
        for idx, row in medicine_data.iterrows():
            priority = 0
            
            # Factor 1: Stock level influence
            if row['Quantity'] > 20:  # High stock
                priority += 3
            elif row['Quantity'] > 10:  # Medium stock
                priority += 1
            elif row['Quantity'] == 0:  # Out of stock (needs stock)
                priority -= 5
            elif row['Quantity'] <= 5:  # Low stock (needs stock)
                priority -= 3
                
            # Factor 2: Expiry urgency (items expiring soon should be moved first)
            if row['DaysUntilExpiry'] <= 30:  # Expiring soon
                priority += 2
            elif row['DaysUntilExpiry'] <= 60:  # Moderate urgency
                priority += 1
                
            # Factor 3: Current status
            if row['StockStatus'] == 'Out of Stock':
                priority -= 4
            elif row['StockStatus'] == 'Low Stock':
                priority -= 2
            elif row['StockStatus'] == 'Expiring Soon':
                priority += 3
                
            medicine_data.loc[idx, 'Priority'] = priority
        
        # Find transfer opportunities
        donors = medicine_data[medicine_data['Priority'] > 0].sort_values('Priority', ascending=False)
        recipients = medicine_data[medicine_data['Priority'] < 0].sort_values('Priority', ascending=True)
        
        # Generate suggestions for this medicine
        for _, donor in donors.iterrows():
            for _, recipient in recipients.iterrows():
                if donor['Store'] != recipient['Store'] and donor['Quantity'] > 5:
                    
                    # Calculate optimal transfer amount
                    max_transfer = min(
                        donor['Quantity'] - 5,  # Leave minimum stock at donor
                        20 - recipient['Quantity']  # Don't exceed reasonable stock level
                    )
                    
                    if max_transfer > 0:
                        # Calculate suggestion score
                        urgency_score = 0
                        if recipient['StockStatus'] == 'Out of Stock':
                            urgency_score = 5
                        elif recipient['StockStatus'] == 'Low Stock':
                            urgency_score = 3
                        elif donor['StockStatus'] == 'Expiring Soon':
                            urgency_score = 4
                            
                        # Impact score based on quantities
                        impact_score = min(max_transfer / 10, 3)  # Normalize to 0-3
                        
                        # Expiry consideration
                        expiry_bonus = 0
                        if donor['DaysUntilExpiry'] <= 30:
                            expiry_bonus = 2
                        elif donor['DaysUntilExpiry'] <= 60:
                            expiry_bonus = 1
                            
                        total_score = urgency_score + impact_score + expiry_bonus
                        
                        suggestion = {
                            'medicine': medicine,
                            'from_store': donor['Store'],
                            'to_store': recipient['Store'],
                            'suggested_quantity': int(max_transfer),
                            'from_current_stock': int(donor['Quantity']),
                            'to_current_stock': int(recipient['Quantity']),
                            'from_status': donor['StockStatus'],
                            'to_status': recipient['StockStatus'],
                            'expiry_date': donor['ExpiryDate'].strftime('%Y-%m-%d'),
                            'days_until_expiry': int(donor['DaysUntilExpiry']),
                            'urgency': 'High' if urgency_score >= 4 else 'Medium' if urgency_score >= 2 else 'Low',
                            'reason': generate_transfer_reason(donor, recipient, max_transfer),
                            'score': round(total_score, 2)
                        }
                        suggestions.append(suggestion)
    
    # Sort by score (highest first) and remove duplicates
    suggestions = sorted(suggestions, key=lambda x: x['score'], reverse=True)
    
    # Remove duplicate suggestions for same medicine between same stores
    seen = set()
    unique_suggestions = []
    for suggestion in suggestions:
        key = (suggestion['medicine'], suggestion['from_store'], suggestion['to_store'])
        if key not in seen:
            seen.add(key)
            unique_suggestions.append(suggestion)
    
    return unique_suggestions[:20]  # Limit to top 20 suggestions

def generate_transfer_reason(donor, recipient, quantity):
    """Generate human-readable reason for transfer suggestion"""
    reasons = []
    
    if recipient['StockStatus'] == 'Out of Stock':
        reasons.append(f"{recipient['Store']} is out of stock")
    elif recipient['StockStatus'] == 'Low Stock':
        reasons.append(f"{recipient['Store']} has low stock ({recipient['Quantity']} units)")
    
    if donor['StockStatus'] == 'Expiring Soon':
        reasons.append(f"items at {donor['Store']} expire in {donor['DaysUntilExpiry']} days")
    elif donor['Quantity'] > 20:
        reasons.append(f"{donor['Store']} has excess stock ({donor['Quantity']} units)")
    
    if not reasons:
        reasons.append("optimize stock distribution")
    
    return f"Transfer {quantity} units to " + " and ".join(reasons)

@app.route('/api/transfer-suggestions')
def transfer_suggestions():
    if inventory_data is None:
        return jsonify({'error': 'No data available. Please upload a CSV file first.'}), 400
    
    suggestions = calculate_transfer_suggestions()
    
    return jsonify({
        'suggestions': suggestions,
        'total_count': len(suggestions),
        'top_3': suggestions[:3] if len(suggestions) >= 3 else suggestions
    })

def calculate_stockout_risk(row, days_horizon=7):
    """Calculate stockout risk percentage for next N days"""
    if row['Quantity'] <= 0:
        return 100.0  # Already out of stock
    
    # Simulate daily consumption rate (simplified model)
    # In real system, this would use historical consumption data
    avg_daily_consumption = max(1, row['Quantity'] / 30)  # Assume 30-day cycle
    
    # Calculate days until stockout
    days_until_stockout = row['Quantity'] / avg_daily_consumption
    
    # Risk increases as we approach stockout within horizon
    if days_until_stockout <= days_horizon:
        return min(100.0, (1 - (days_until_stockout / days_horizon)) * 100)
    
    # Also consider expiry risk
    if row['DaysUntilExpiry'] <= days_horizon and row['DaysUntilExpiry'] > 0:
        expiry_risk = (1 - (row['DaysUntilExpiry'] / days_horizon)) * 50
        return min(100.0, expiry_risk)
    
    return 0.0

def generate_store_coordinates(store_name):
    """Generate realistic coordinates for stores (for demonstration)"""
    # In real system, this would come from actual store location data
    hash_val = hash(store_name)
    
    # Simulate stores distributed across major regions
    regions = [
        {"lat": 40.7128, "lng": -74.0060, "name": "Northeast"},  # NYC area
        {"lat": 34.0522, "lng": -118.2437, "name": "West"},     # LA area
        {"lat": 41.8781, "lng": -87.6298, "name": "Midwest"},   # Chicago area
        {"lat": 29.7604, "lng": -95.3698, "name": "South"},     # Houston area
        {"lat": 47.6062, "lng": -122.3321, "name": "Northwest"} # Seattle area
    ]
    
    region = regions[abs(hash_val) % len(regions)]
    
    # Add random offset within region
    lat_offset = (hash_val % 1000) / 10000.0 - 0.05  # ±0.05 degrees
    lng_offset = ((hash_val // 1000) % 1000) / 10000.0 - 0.05
    
    return {
        "lat": region["lat"] + lat_offset,
        "lng": region["lng"] + lng_offset,
        "region": region["name"]
    }

def create_hexagonal_clusters(stores_data, hex_resolution=8):
    """Create hexagonal clusters for risk mapping"""
    # For demonstration, we'll create a simplified hex grid
    # In production, use H3 library for proper hexagonal indexing
    
    clusters = []
    hex_size = 0.1  # Degrees
    
    # Group stores by approximate hex regions
    hex_groups = {}
    
    for store in stores_data:
        # Simplified hex coordinate calculation
        hex_lat = round(store['lat'] / hex_size) * hex_size
        hex_lng = round(store['lng'] / hex_size) * hex_size
        hex_key = f"{hex_lat},{hex_lng}"
        
        if hex_key not in hex_groups:
            hex_groups[hex_key] = {
                'lat': hex_lat,
                'lng': hex_lng,
                'stores': [],
                'total_risk': 0,
                'store_count': 0
            }
        
        hex_groups[hex_key]['stores'].append(store)
        hex_groups[hex_key]['total_risk'] += store['risk_score']
        hex_groups[hex_key]['store_count'] += 1
    
    # Calculate average risk per hex
    for hex_key, hex_data in hex_groups.items():
        avg_risk = hex_data['total_risk'] / hex_data['store_count']
        clusters.append({
            'hex_id': hex_key,
            'lat': hex_data['lat'],
            'lng': hex_data['lng'],
            'avg_risk': round(avg_risk, 2),
            'store_count': hex_data['store_count'],
            'risk_category': 'High' if avg_risk > 70 else 'Medium' if avg_risk > 30 else 'Low',
            'stores': hex_data['stores'][:5]  # Top 5 risky stores in hex
        })
    
    return sorted(clusters, key=lambda x: x['avg_risk'], reverse=True)


@app.route('/api/risk-matrix-heatmap')
def risk_matrix_heatmap():
    if inventory_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    view_type = request.args.get('view', 'category')  # 'category' or 'medicine'
    
    # Calculate risk metrics
    risk_data = inventory_data.copy()
    risk_data['RiskScore'] = risk_data.apply(
        lambda row: calculate_stockout_risk(row, 7), axis=1
    )
    
    # Add regions to stores
    store_regions = {}
    for store in risk_data['Store'].unique():
        coords = generate_store_coordinates(store)
        store_regions[store] = coords['region']
    
    risk_data['Region'] = risk_data['Store'].map(store_regions)
    
    if view_type == 'category' and 'Category' in risk_data.columns:
        # Group by category for better visualization with large datasets
        risk_matrix = risk_data.groupby(['Region', 'Category']).agg({
            'RiskScore': 'mean',
            'DaysUntilExpiry': 'mean',
            'Quantity': 'sum'
        }).reset_index()
        
        # Pivot for heatmap
        risk_pivot = risk_matrix.pivot(
            index='Region', 
            columns='Category', 
            values='RiskScore'
        ).fillna(0)
        
        expiry_pivot = risk_matrix.pivot(
            index='Region', 
            columns='Category', 
            values='DaysUntilExpiry'
        ).fillna(0)
        
        quantity_pivot = risk_matrix.pivot(
            index='Region', 
            columns='Category', 
            values='Quantity'
        ).fillna(0)
        
        column_label = 'Category'
        
    else:
        # Original medicine-level view (for smaller datasets or detailed analysis)
        # Limit to top 50 highest-risk medicines for visualization
        top_risk_medicines = risk_data.groupby('Medicine')['RiskScore'].mean().nlargest(50).index
        risk_data_filtered = risk_data[risk_data['Medicine'].isin(top_risk_medicines)]
        
        risk_matrix = risk_data_filtered.groupby(['Region', 'Medicine']).agg({
            'RiskScore': 'mean',
            'DaysUntilExpiry': 'mean',
            'Quantity': 'sum'
        }).reset_index()
        
        # Pivot for heatmap
        risk_pivot = risk_matrix.pivot(
            index='Region', 
            columns='Medicine', 
            values='RiskScore'
        ).fillna(0)
        
        expiry_pivot = risk_matrix.pivot(
            index='Region', 
            columns='Medicine', 
            values='DaysUntilExpiry'
        ).fillna(0)
        
        quantity_pivot = risk_matrix.pivot(
            index='Region', 
            columns='Medicine', 
            values='Quantity'
        ).fillna(0)
        
        column_label = 'Medicine (Top 50 by Risk)'
    
    return jsonify({
        'risk_matrix': {
            'regions': risk_pivot.index.tolist(),
            'columns': risk_pivot.columns.tolist(),
            'risk_scores': risk_pivot.values.tolist(),
            'column_label': column_label
        },
        'expiry_matrix': {
            'regions': expiry_pivot.index.tolist(),
            'columns': expiry_pivot.columns.tolist(),
            'expiry_days': expiry_pivot.values.tolist(),
            'column_label': column_label
        },
        'quantity_matrix': {
            'regions': quantity_pivot.index.tolist(),
            'columns': quantity_pivot.columns.tolist(),
            'quantities': quantity_pivot.values.tolist(),
            'column_label': column_label
        }
    })

@app.route('/api/charts/stock-aging')
def stock_aging_timeline():
    global inventory_data
    if inventory_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Calculate aging buckets based on days until expiry
        aging_data = []
        
        for _, row in inventory_data.iterrows():
            days_until_expiry = row['DaysUntilExpiry']
            
            if days_until_expiry < 0:
                bucket = 'Expired'
            elif days_until_expiry <= 30:
                bucket = '0-30 Days'
            elif days_until_expiry <= 90:
                bucket = '31-90 Days'
            elif days_until_expiry <= 180:
                bucket = '91-180 Days'
            elif days_until_expiry <= 365:
                bucket = '181-365 Days'
            else:
                bucket = '365+ Days'
            
            aging_data.append({
                'Store': row['Store'],
                'Medicine': row['Medicine'],
                'Category': row['Category'],
                'Quantity': row['Quantity'],
                'DaysUntilExpiry': days_until_expiry,
                'AgingBucket': bucket,
                'Value': row['Quantity'] * 10  # Approximate value
            })
        
        aging_df = pd.DataFrame(aging_data)
        
        # Create timeline visualization data
        bucket_order = ['Expired', '0-30 Days', '31-90 Days', '91-180 Days', '181-365 Days', '365+ Days']
        bucket_colors = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e', '#06b6d4']
        
        timeline_data = []
        for i, bucket in enumerate(bucket_order):
            bucket_data = aging_df[aging_df['AgingBucket'] == bucket]
            if not bucket_data.empty:
                timeline_data.append({
                    'bucket': bucket,
                    'quantity': int(bucket_data['Quantity'].sum()),
                    'value': int(bucket_data['Value'].sum()),
                    'count': len(bucket_data),
                    'color': bucket_colors[i]
                })
        
        return jsonify({
            'timeline_data': timeline_data,
            'total_value_at_risk': int(aging_df[aging_df['DaysUntilExpiry'] <= 90]['Value'].sum())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/stockout-risk')
def stockout_risk_forecast():
    global inventory_data
    if inventory_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Calculate stockout risk based on current stock levels and consumption patterns
        risk_data = []
        
        for _, row in inventory_data.iterrows():
            quantity = row['Quantity']
            days_until_expiry = row['DaysUntilExpiry']
            
            # Simulate daily consumption rate
            category_consumption = {
                'Analgesics': 5, 'Antibiotics': 8, 'Vitamins': 3, 'Antacids': 4,
                'Cold & Flu': 6, 'Diabetes': 7, 'Heart': 5, 'Blood Pressure': 6,
                'Cholesterol': 4, 'Respiratory': 5, 'Supplements': 2, 'Skincare': 3,
                'Digestive': 4, 'Mental Health': 3, 'Hormonal': 4, 'Eye Care': 2,
                'Emergency': 8
            }
            
            daily_consumption = category_consumption.get(row['Category'], 4)
            days_of_stock = quantity / daily_consumption if daily_consumption > 0 else 999
            
            if days_of_stock <= 7:
                risk_level = 'Critical'
                risk_score = 90
            elif days_of_stock <= 14:
                risk_level = 'High'
                risk_score = 70
            elif days_of_stock <= 30:
                risk_level = 'Medium'
                risk_score = 40
            else:
                risk_level = 'Low'
                risk_score = 10
            
            risk_data.append({
                'Store': row['Store'],
                'Medicine': row['Medicine'],
                'Category': row['Category'],
                'CurrentStock': quantity,
                'DaysOfStock': round(days_of_stock, 1),
                'RiskLevel': risk_level,
                'RiskScore': risk_score
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        # Aggregate by risk level
        risk_summary = risk_df.groupby('RiskLevel').agg({
            'Medicine': 'count',
            'CurrentStock': 'sum'
        }).reset_index()
        
        return jsonify({
            'risk_summary': risk_summary.to_dict('records'),
            'critical_items': risk_df[risk_df['RiskLevel'] == 'Critical'].to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/transfer-opportunities')
def transfer_opportunity_map():
    global inventory_data
    if inventory_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Calculate transfer opportunities by finding surplus vs deficit stores
        opportunities = []
        
        # Group by medicine to find surplus/deficit patterns
        medicine_groups = inventory_data.groupby('Medicine')
        
        for medicine, group in medicine_groups:
            stores_data = []
            
            for _, row in group.iterrows():
                quantity = row['Quantity']
                ideal_stock = 100  # Simplified ideal stock level
                surplus_deficit = quantity - ideal_stock
                
                stores_data.append({
                    'Store': row['Store'],
                    'Region': row['Region'],
                    'CurrentStock': quantity,
                    'SurplusDeficit': surplus_deficit,
                    'Category': row['Category']
                })
            
            # Find transfer opportunities
            stores_df = pd.DataFrame(stores_data)
            surplus_stores = stores_df[stores_df['SurplusDeficit'] > 20].sort_values('SurplusDeficit', ascending=False)
            deficit_stores = stores_df[stores_df['SurplusDeficit'] < -10].sort_values('SurplusDeficit')
            
            # Create transfer suggestions
            for _, surplus_store in surplus_stores.iterrows():
                for _, deficit_store in deficit_stores.iterrows():
                    if surplus_store['Store'] != deficit_store['Store']:
                        transfer_quantity = min(
                            surplus_store['SurplusDeficit'] * 0.7,
                            abs(deficit_store['SurplusDeficit']) * 0.8
                        )
                        
                        if transfer_quantity >= 10:
                            opportunities.append({
                                'Medicine': medicine,
                                'FromStore': surplus_store['Store'],
                                'FromRegion': surplus_store['Region'],
                                'ToStore': deficit_store['Store'],
                                'ToRegion': deficit_store['Region'],
                                'TransferQuantity': round(transfer_quantity),
                                'EstimatedSavings': round(transfer_quantity * 10),
                                'Category': surplus_store['Category']
                            })
        
        opportunities_df = pd.DataFrame(opportunities)
        if not opportunities_df.empty:
            opportunities_df = opportunities_df.sort_values('EstimatedSavings', ascending=False)
        
        return jsonify({
            'opportunities': opportunities_df.head(20).to_dict('records') if not opportunities_df.empty else [],
            'total_savings_potential': int(opportunities_df['EstimatedSavings'].sum()) if not opportunities_df.empty else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/risk-matrix')
def risk_matrix_chart():
    if inventory_data is None:
        return jsonify({'error': 'No data available'}), 400
    
    view_type = request.args.get('view', 'category')
    
    # Get matrix data
    matrix_response = risk_matrix_heatmap()
    matrix_data = json.loads(matrix_response.data)
    
    risk_matrix = matrix_data['risk_matrix']
    
    fig = go.Figure(data=go.Heatmap(
        z=risk_matrix['risk_scores'],
        x=risk_matrix['columns'],
        y=risk_matrix['regions'],
        colorscale=[
            [0, '#00ff88'],     # Low risk - green
            [0.3, '#ffd700'],   # Medium risk - yellow
            [0.7, '#ff8c00'],   # High risk - orange
            [1, '#ff0000']      # Critical risk - red
        ],
        hoverongaps=False,
        hovertemplate='<b>%{y} - %{x}</b><br>Risk Score: %{z:.1f}%<extra></extra>',
        colorbar=dict(
            title="Risk Score (%)",
            titleside="right"
        )
    ))
    
    title = f"Risk Matrix: Regions vs {risk_matrix['column_label']}"
    
    fig.update_layout(
        title=title,
        xaxis_title=risk_matrix['column_label'],
        yaxis_title='Regions',
        template='plotly_dark',
        paper_bgcolor='#111827',
        plot_bgcolor='#111827',
        font=dict(color='white', size=12),
        height=600,
        xaxis=dict(tickangle=45) if view_type == 'medicine' else dict()
    )
    
    return jsonify(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

# ============================================================================
# MACHINE LEARNING API ENDPOINTS
# ============================================================================

@app.route('/api/ml/train-models', methods=['POST'])
def train_ml_models():
    """Train ML models on current inventory data"""
    if not ML_ENABLED or ml_manager is None:
        return jsonify({'error': 'ML functionality not available'}), 400
    
    if inventory_data is None:
        return jsonify({'error': 'No inventory data available for training'}), 400
    
    try:
        # Train models
        trained_count = ml_manager.train_initial_models(inventory_data)
        
        # Get model statistics
        stats = ml_manager.get_model_stats()
        
        return jsonify({
            'message': f'✅ Successfully trained {trained_count} ML models',
            'trained_models': trained_count,
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to train models: {str(e)}'}), 500

@app.route('/api/ml/demand-forecast')
def get_demand_forecast():
    """Get demand forecasts for all store×medicine combinations"""
    if not ML_ENABLED or ml_manager is None:
        return jsonify({'error': 'ML functionality not available'}), 400
    
    if inventory_data is None:
        return jsonify({'error': 'No inventory data available'}), 400
    
    try:
        # Get predictions
        predictions = ml_manager.get_predictions(inventory_data)
        
        # Convert to serializable format
        forecast_data = []
        for pred in predictions:
            forecast_data.append({
                'store_id': pred.store_id,
                'medicine': pred.medicine,
                'predicted_demand_7d': round(pred.predicted_demand_7d, 2),
                'predicted_demand_14d': round(pred.predicted_demand_14d, 2),
                'confidence_interval_low': round(pred.confidence_interval_low, 2),
                'confidence_interval_high': round(pred.confidence_interval_high, 2),
                'stockout_probability_7d': round(pred.stockout_probability_7d, 3),
                'stockout_probability_14d': round(pred.stockout_probability_14d, 3),
                'days_of_cover': round(pred.days_of_cover, 1) if pred.days_of_cover != float('inf') else 999,
                'forecast_method': pred.forecast_method
            })
        
        # Aggregate statistics
        high_risk_count = sum(1 for p in predictions if p.stockout_probability_7d > 0.7)
        medium_risk_count = sum(1 for p in predictions if 0.3 < p.stockout_probability_7d <= 0.7)
        
        return jsonify({
            'forecasts': forecast_data,
            'summary': {
                'total_predictions': len(predictions),
                'high_risk_stockouts': high_risk_count,
                'medium_risk_stockouts': medium_risk_count,
                'avg_days_of_cover': round(np.mean([p.days_of_cover for p in predictions if p.days_of_cover != float('inf')]), 1)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate forecasts: {str(e)}'}), 500

@app.route('/api/ml/transfer-optimization')
def get_ml_transfer_suggestions():
    """Get ML-optimized transfer suggestions"""
    if not ML_ENABLED or ml_manager is None:
        return jsonify({'error': 'ML functionality not available'}), 400
    
    if inventory_data is None:
        return jsonify({'error': 'No inventory data available'}), 400
    
    try:
        # Generate store locations (using the same method as global risk map)
        store_locations = {}
        for store in inventory_data['Store'].unique():
            coords = generate_store_coordinates(store)
            store_locations[store] = {'lat': coords['lat'], 'lng': coords['lng']}
        
        # Get transfer suggestions
        suggestions = ml_manager.get_transfer_suggestions(inventory_data, store_locations)
        
        # Convert to serializable format
        suggestion_data = []
        for sugg in suggestions:
            suggestion_data.append({
                'from_store': sugg.from_store,
                'to_store': sugg.to_store,
                'medicine': sugg.medicine,
                'quantity': sugg.quantity,
                'urgency_score': round(sugg.urgency_score, 2),
                'cost_savings': round(sugg.cost_savings, 2),
                'transfer_reason': sugg.transfer_reason,
                'expected_delivery_days': sugg.expected_delivery_days
            })
        
        return jsonify({
            'ml_transfer_suggestions': suggestion_data,
            'summary': {
                'total_suggestions': len(suggestions),
                'high_urgency': sum(1 for s in suggestions if s.urgency_score > 70),
                'total_units': sum(s.quantity for s in suggestions),
                'estimated_savings': sum(s.cost_savings for s in suggestions)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate transfer suggestions: {str(e)}'}), 500

@app.route('/api/ml/update-models', methods=['POST'])
def update_ml_models():
    """Update ML models with new sales data"""
    if not ML_ENABLED or ml_manager is None:
        return jsonify({'error': 'ML functionality not available'}), 400
    
    try:
        # Get new sales data from request
        new_data = request.get_json()
        
        if not new_data or 'sales_data' not in new_data:
            return jsonify({'error': 'Sales data required in request body'}), 400
        
        # Update models
        updated_count = ml_manager.update_with_new_data(new_data['sales_data'])
        
        return jsonify({
            'message': f'✅ Updated {updated_count} models with new data',
            'updated_models': updated_count
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to update models: {str(e)}'}), 500

@app.route('/api/ml/model-stats')
def get_ml_model_stats():
    """Get ML model statistics and status"""
    if not ML_ENABLED or ml_manager is None:
        return jsonify({'error': 'ML functionality not available'}), 400
    
    try:
        stats = ml_manager.get_model_stats()
        
        return jsonify({
            'ml_enabled': True,
            'model_statistics': stats,
            'status': 'operational' if stats['total_models'] > 0 else 'no_models_trained'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model stats: {str(e)}'}), 500

@app.route('/api/ml/stockout-risk-analysis')
def stockout_risk_analysis():
    """Advanced stockout risk analysis using ML predictions"""
    if not ML_ENABLED or ml_manager is None:
        return jsonify({'error': 'ML functionality not available'}), 400
    
    if inventory_data is None:
        return jsonify({'error': 'No inventory data available'}), 400
    
    try:
        # Get predictions
        predictions = ml_manager.get_predictions(inventory_data)
        
        # Analyze risk by category and region
        risk_by_category = {}
        risk_by_store = {}
        
        for pred in predictions:
            # Find category and region from inventory data
            item_info = inventory_data[
                (inventory_data['Store'] == pred.store_id) & 
                (inventory_data['Medicine'] == pred.medicine)
            ]
            
            if not item_info.empty:
                category = item_info.iloc[0].get('Category', 'Unknown')
                region = item_info.iloc[0].get('Region', 'Unknown')
                
                # Risk by category
                if category not in risk_by_category:
                    risk_by_category[category] = []
                risk_by_category[category].append(pred.stockout_probability_7d)
                
                # Risk by store
                if pred.store_id not in risk_by_store:
                    risk_by_store[pred.store_id] = []
                risk_by_store[pred.store_id].append(pred.stockout_probability_7d)
        
        # Calculate aggregated risks
        category_risks = {
            cat: {
                'avg_risk': round(np.mean(risks), 3),
                'max_risk': round(np.max(risks), 3),
                'items_at_risk': sum(1 for r in risks if r > 0.5)
            }
            for cat, risks in risk_by_category.items()
        }
        
        store_risks = {
            store: {
                'avg_risk': round(np.mean(risks), 3),
                'max_risk': round(np.max(risks), 3),
                'items_at_risk': sum(1 for r in risks if r > 0.5)
            }
            for store, risks in risk_by_store.items()
        }
        
        return jsonify({
            'risk_analysis': {
                'by_category': category_risks,
                'by_store': store_risks,
                'overall': {
                    'total_items_analyzed': len(predictions),
                    'high_risk_items': sum(1 for p in predictions if p.stockout_probability_7d > 0.7),
                    'medium_risk_items': sum(1 for p in predictions if 0.3 < p.stockout_probability_7d <= 0.7),
                    'avg_global_risk': round(np.mean([p.stockout_probability_7d for p in predictions]), 3)
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to analyze stockout risk: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
