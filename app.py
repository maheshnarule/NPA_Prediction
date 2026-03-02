


import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import os
from datetime import datetime
import logging
import json
from math import ceil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

# Global variables
model = None
scaler = None
feature_names = None
MAX_LOAN_TERM_MONTHS = 300  # Maximum 25 years (300 months)

def init_app():
    """Initialize the application and load models"""
    global model, scaler, feature_names
    
    try:
        # Load model
        with open('model/npa_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature names
        with open('model/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Initialize on startup
init_app()

def preprocess_input(form_data):
    """Preprocess form data for prediction"""
    try:
        # Create a dictionary with default values for all features
        input_dict = {}
        
        # Convert monthly income to annual income for the model
        monthly_income = float(form_data.get('MonthlyIncome', 0))
        annual_income = monthly_income * 12
        
        # Store both values for display
        input_dict['MonthlyIncome'] = monthly_income
        input_dict['AnnualIncome'] = annual_income
        
        # Numerical features - use annual income for the model
        numerical_features = ['Age', 'LoanAmount', 'CreditScore', 
                             'MonthsEmployed', 'NumCreditLines', 'InterestRate', 
                             'LoanTerm', 'DTIRatio']
        
        for feat in numerical_features:
            input_dict[feat] = float(form_data.get(feat, 0))
        
        # Add annual income (converted from monthly)
        input_dict['Income'] = annual_income
        
        # Categorical features - one-hot encoding
        categorical_mappings = {
            'EmploymentType': ['EmploymentType_Full-time', 'EmploymentType_Part-time',
                              'EmploymentType_Self-employed', 'EmploymentType_Unemployed'],
            'MaritalStatus': ['MaritalStatus_Divorced', 'MaritalStatus_Married', 
                            'MaritalStatus_Single'],
            'HasMortgage': ['HasMortgage_Yes'],
            'HasDependents': ['HasDependents_Yes'],
            'LoanPurpose': ['LoanPurpose_Auto', 'LoanPurpose_Business', 
                           'LoanPurpose_Education', 'LoanPurpose_Home', 
                           'LoanPurpose_Other'],
            'HasCoSigner': ['HasCoSigner_Yes']
        }
        
        # Initialize all categorical features to 0
        for categories in categorical_mappings.values():
            for cat in categories:
                input_dict[cat] = 0
        
        # Set the appropriate category to 1 based on form data
        # Employment Type
        employment = form_data.get('EmploymentType', '')
        if employment == 'Full-time':
            input_dict['EmploymentType_Full-time'] = 1
        elif employment == 'Part-time':
            input_dict['EmploymentType_Part-time'] = 1
        elif employment == 'Self-employed':
            input_dict['EmploymentType_Self-employed'] = 1
        elif employment == 'Unemployed':
            input_dict['EmploymentType_Unemployed'] = 1
        
        # Marital Status
        marital = form_data.get('MaritalStatus', '')
        if marital == 'Divorced':
            input_dict['MaritalStatus_Divorced'] = 1
        elif marital == 'Married':
            input_dict['MaritalStatus_Married'] = 1
        elif marital == 'Single':
            input_dict['MaritalStatus_Single'] = 1
        
        # Binary features
        if form_data.get('HasMortgage') == 'Yes':
            input_dict['HasMortgage_Yes'] = 1
        
        if form_data.get('HasDependents') == 'Yes':
            input_dict['HasDependents_Yes'] = 1
        
        if form_data.get('HasCoSigner') == 'Yes':
            input_dict['HasCoSigner_Yes'] = 1
        
        # Loan Purpose
        purpose = form_data.get('LoanPurpose', '')
        if purpose == 'Auto':
            input_dict['LoanPurpose_Auto'] = 1
        elif purpose == 'Business':
            input_dict['LoanPurpose_Business'] = 1
        elif purpose == 'Education':
            input_dict['LoanPurpose_Education'] = 1
        elif purpose == 'Home':
            input_dict['LoanPurpose_Home'] = 1
        elif purpose == 'Other':
            input_dict['LoanPurpose_Other'] = 1
        
        # Create DataFrame for model input
        df = pd.DataFrame([input_dict])
        
        # Ensure all feature names are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training
        df = df[feature_names]
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        return scaled_features, input_dict
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise e

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    elif request.method == 'POST':
        try:
            # Get form data
            form_data = request.form.to_dict()
            
            # Validate required fields
            required_fields = ['Age', 'MonthlyIncome', 'LoanAmount', 'CreditScore', 
                              'MonthsEmployed', 'DTIRatio', 'EmploymentType', 
                              'MaritalStatus', 'LoanNumber', 'CustomerName',
                              'HasDependents', 'HasMortgage', 'NumCreditLines',
                              'InterestRate', 'LoanTerm', 'LoanPurpose', 'HasCoSigner']
            
            for field in required_fields:
                if not form_data.get(field):
                    return render_template('predict.html', 
                                         error=f"Missing required field: {field}")
            
            # Additional validation
            try:
                age = int(form_data.get('Age', 0))
                if age < 18 or age > 100:
                    return render_template('predict.html', 
                                         error="Age must be between 18 and 100")
                
                monthly_income = float(form_data.get('MonthlyIncome', 0))
                if monthly_income < 0:
                    return render_template('predict.html',
                                         error="Monthly income cannot be negative")
                
                dti = float(form_data.get('DTIRatio', 0))
                if dti < 0.01 or dti > 1.0:
                    return render_template('predict.html',
                                         error="DTI Ratio must be between 0.01 and 1.0")
                
                credit_score = int(form_data.get('CreditScore', 0))
                if credit_score < 650 or credit_score > 850:
                    return render_template('predict.html',
                                         error="Credit Score must be between 650 and 850")
                
                loan_term = int(form_data.get('LoanTerm', 0))
                if loan_term < 1 or loan_term > MAX_LOAN_TERM_MONTHS:
                    return render_template('predict.html',
                                         error=f"Loan Term must be between 1 and {MAX_LOAN_TERM_MONTHS} months")
                
            except ValueError as e:
                return render_template('predict.html',
                                     error=f"Invalid input format: {str(e)}")
            
            # Preprocess and predict
            scaled_features, raw_features = preprocess_input(form_data)
            
            if model is None:
                return render_template('predict.html', 
                                     error="Model not loaded. Please contact administrator.")
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0]
            
            # Prepare result
            default_probability = probability[1] * 100  # Probability of default
            risk_score = default_probability
            
            # Determine risk level
            if risk_score < 20:
                risk_level = 'Low'
                risk_color = 'success'
                recommendation = "Loan application looks favorable"
            elif risk_score < 50:
                risk_level = 'Medium'
                risk_color = 'warning'
                recommendation = "Consider additional verification"
            else:
                risk_level = 'High'
                risk_color = 'danger'
                recommendation = "High risk - recommend rejection or additional collateral"
            
            # Generate specific recommendations
            recommendations = generate_recommendations(form_data, risk_score, raw_features)
            
            # Generate manager suggestions for high-risk customers
            manager_suggestions = []
            if risk_score > 50:  # High or Very High risk
                manager_suggestions = generate_manager_suggestions(form_data, raw_features, risk_score)
            
            result = {
                'prediction': 'High Risk (Potential NPA)' if prediction == 1 else 'Low Risk (Good Loan)',
                'risk_score': round(risk_score, 2),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'default_probability': round(default_probability, 2),
                'confidence': round(max(probability) * 100, 2),
                'recommendation': recommendation,
                'specific_recommendations': recommendations,
                'manager_suggestions': manager_suggestions,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'probability': probability[1],
                'annual_income': raw_features.get('AnnualIncome', 0),
                'is_high_risk': risk_score > 50
            }
            
            # Store in session for result page
            session['prediction_result'] = result
            session['input_data'] = form_data
            session['raw_features'] = raw_features
            
            return redirect(url_for('result'))
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return render_template('predict.html', 
                                 error=f"Error processing request: {str(e)}")

@app.route('/result')
def result():
    """Result page"""
    result = session.get('prediction_result')
    input_data = session.get('input_data')
    raw_features = session.get('raw_features')
    
    if not result or not input_data:
        return redirect(url_for('predict'))
    
    # Ensure all required fields are in input_data for the template
    # Convert monthly to annual if not already done
    if 'MonthlyIncome' in input_data and 'AnnualIncome' not in input_data:
        try:
            monthly_income = float(input_data.get('MonthlyIncome', 0))
            input_data['AnnualIncome'] = monthly_income * 12
        except:
            input_data['AnnualIncome'] = 0
    
    # Also check raw_features
    if raw_features:
        if 'MonthlyIncome' in raw_features:
            input_data['MonthlyIncome'] = raw_features.get('MonthlyIncome')
        if 'AnnualIncome' in raw_features:
            input_data['AnnualIncome'] = raw_features.get('AnnualIncome')
    
    # Format DTI Ratio for display
    if 'DTIRatio' in input_data:
        try:
            input_data['DTIRatio'] = float(input_data['DTIRatio'])
        except:
            pass
    
    return render_template('result.html', result=result, input_data=input_data)

def generate_manager_suggestions(form_data, raw_features, risk_score):
    """Generate manager-level suggestions for high-risk customers"""
    suggestions = []
    
    # Get current values
    current_loan_amount = float(form_data.get('LoanAmount', 0))
    current_interest_rate = float(form_data.get('InterestRate', 0))
    current_loan_term = int(form_data.get('LoanTerm', 0))
    monthly_income = float(form_data.get('MonthlyIncome', 0))
    
    # Calculate current monthly payment
    current_monthly_payment = calculate_monthly_payment(
        current_loan_amount, 
        current_interest_rate, 
        current_loan_term
    )
    
    # Suggestion 1: Increase loan tenure (based on risk parameters)
    if current_loan_term < MAX_LOAN_TERM_MONTHS:
        # Determine appropriate tenure extensions based on risk score and current term
        tenure_suggestions = get_tenure_suggestions(current_loan_term, risk_score)
        
        for suggestion in tenure_suggestions:
            new_term = suggestion['new_term']
            new_payment = calculate_monthly_payment(current_loan_amount, current_interest_rate, new_term)
            payment_reduction = current_monthly_payment - new_payment
            reduction_percentage = (payment_reduction / current_monthly_payment) * 100 if current_monthly_payment > 0 else 0
            
            if payment_reduction > 0:
                suggestions.append({
                    'type': 'tenure_extension',
                    'title': 'Increase Loan Tenure',
                    'current_term': f"{current_loan_term} months",
                    'suggested_term': f"{new_term} months",
                    'current_payment': f"₹{current_monthly_payment:,.2f}",
                    'new_payment': f"₹{new_payment:,.2f}",
                    'payment_reduction': f"₹{payment_reduction:,.2f} ({reduction_percentage:.1f}%)",
                    'payment_to_income': f"{(new_payment / monthly_income) * 100:.1f}%" if monthly_income > 0 else "N/A",
                    'description': suggestion['description']
                })
    
    # Suggestion 2: Adjust interest rate for very high-risk customers
    if risk_score > 70:  # Very high risk
        # Determine appropriate interest rate increase based on risk
        rate_increase = get_interest_rate_increase(risk_score, current_interest_rate)
        new_rate = current_interest_rate + rate_increase
        
        if new_rate <= 30:  # Maximum interest rate
            new_payment = calculate_monthly_payment(current_loan_amount, new_rate, current_loan_term)
            payment_increase = new_payment - current_monthly_payment
            increase_percentage = (payment_increase / current_monthly_payment) * 100 if current_monthly_payment > 0 else 0
            additional_total_interest = (new_payment * current_loan_term) - (current_monthly_payment * current_loan_term)
            
            suggestions.append({
                'type': 'interest_adjustment',
                'title': 'Adjust Interest Rate',
                'current_rate': f"{current_interest_rate}%",
                'suggested_rate': f"{new_rate}%",
                'rate_increase': f"{rate_increase}%",
                'current_payment': f"₹{current_monthly_payment:,.2f}",
                'new_payment': f"₹{new_payment:,.2f}",
                'payment_increase': f"₹{payment_increase:,.2f} ({increase_percentage:.1f}%)",
                'additional_total_interest': f"₹{additional_total_interest:,.2f}",
                'description': f"Increase interest rate from {current_interest_rate}% to {new_rate}% to compensate for higher risk"
            })
    
    # Suggestion 3: Reduce loan amount based on payment-to-income ratio
    current_payment_to_income = (current_monthly_payment / monthly_income) * 100 if monthly_income > 0 else 0
    
    if current_payment_to_income > 35:  # If payment is more than 35% of income
        # Target payment-to-income ratio (25-30%)
        target_pti = 30 if risk_score > 60 else 25
        target_monthly_payment = monthly_income * (target_pti / 100)
        
        if target_monthly_payment < current_monthly_payment:
            # Calculate new loan amount that achieves target payment
            # This is an approximation - we'd need to solve the amortization formula
            # For simplicity, we'll suggest proportional reduction
            reduction_needed = (current_monthly_payment - target_monthly_payment) / current_monthly_payment
            suggested_reduction_percent = min(40, reduction_needed * 100)  # Cap at 40% reduction
            new_amount = current_loan_amount * (1 - suggested_reduction_percent/100)
            
            # Recalculate with exact new amount
            new_payment = calculate_monthly_payment(new_amount, current_interest_rate, current_loan_term)
            payment_reduction = current_monthly_payment - new_payment
            
            suggestions.append({
                'type': 'amount_reduction',
                'title': 'Reduce Loan Amount',
                'current_amount': f"₹{current_loan_amount:,.2f}",
                'suggested_amount': f"₹{new_amount:,.2f}",
                'amount_reduction': f"₹{current_loan_amount - new_amount:,.2f} ({suggested_reduction_percent:.1f}%)",
                'current_payment': f"₹{current_monthly_payment:,.2f}",
                'new_payment': f"₹{new_payment:,.2f}",
                'payment_reduction': f"₹{payment_reduction:,.2f}",
                'description': f"Reduce loan amount by {suggested_reduction_percent:.1f}% to achieve target payment-to-income ratio of {target_pti}%"
            })
    
    # Suggestion 4: Require additional collateral or co-signer
    if form_data.get('HasCoSigner') == 'No' or risk_score > 60:
        collateral_needed = calculate_collateral_needed(current_loan_amount, risk_score)
        
        suggestions.append({
            'type': 'security_requirement',
            'title': 'Additional Security Required',
            'description': 'Require additional collateral or co-signer',
            'current_status': 'No co-signer' if form_data.get('HasCoSigner') == 'No' else 'Insufficient security',
            'recommendation': f'Mandatory: Provide collateral worth ₹{collateral_needed:,.2f} or add creditworthy co-signer',
            'collateral_amount': f"₹{collateral_needed:,.2f}",
            'collateral_percentage': f"{calculate_collateral_percentage(risk_score):.1f}%"
        })
    
    # Suggestion 5: Staged disbursement for very high risk
    if risk_score > 75 and current_loan_amount > 500000:  # Large loans with very high risk
        stages = get_disbursement_stages(current_loan_amount, risk_score)
        
        suggestions.append({
            'type': 'staged_disbursement',
            'title': 'Staged Disbursement',
            'description': 'Release funds in stages based on milestones',
            'current_amount': f"₹{current_loan_amount:,.2f}",
            'stages': stages,
            'recommendation': 'Release funds in stages with performance milestones'
        })
    
    return suggestions

def get_tenure_suggestions(current_term, risk_score):
    """Get appropriate tenure extension suggestions based on parameters"""
    suggestions = []
    
    # Define tenure extension options based on risk and current term
    if risk_score <= 60:
        # Moderate risk - suggest moderate extensions
        if current_term <= 60:  # Up to 5 years
            suggestions.append({
                'new_term': min(120, MAX_LOAN_TERM_MONTHS),  # Extend to 10 years max
                'description': f"Extend from {current_term} months to 120 months to reduce EMI burden"
            })
        elif current_term <= 120:  # 5-10 years
            suggestions.append({
                'new_term': min(180, MAX_LOAN_TERM_MONTHS),  # Extend to 15 years max
                'description': f"Extend from {current_term} months to 180 months to improve affordability"
            })
        elif current_term <= 180:  # 10-15 years
            suggestions.append({
                'new_term': min(240, MAX_LOAN_TERM_MONTHS),  # Extend to 20 years max
                'description': f"Extend from {current_term} months to 240 months for better cash flow"
            })
        elif current_term <= 240:  # 15-20 years
            suggestions.append({
                'new_term': MAX_LOAN_TERM_MONTHS,  # Extend to max
                'description': f"Extend from {current_term} months to {MAX_LOAN_TERM_MONTHS} months for maximum affordability"
            })
    else:
        # High risk - suggest more conservative extensions
        if current_term <= 36:  # Up to 3 years
            suggestions.append({
                'new_term': min(60, MAX_LOAN_TERM_MONTHS),  # Extend to 5 years max
                'description': f"Extend from {current_term} months to 60 months for high-risk applicant"
            })
        elif current_term <= 60:  # 3-5 years
            suggestions.append({
                'new_term': min(84, MAX_LOAN_TERM_MONTHS),  # Extend to 7 years max
                'description': f"Extend from {current_term} months to 84 months with monitoring"
            })
        elif current_term <= 84:  # 5-7 years
            suggestions.append({
                'new_term': min(120, MAX_LOAN_TERM_MONTHS),  # Extend to 10 years max
                'description': f"Extend from {current_term} months to 120 months for high-risk management"
            })
        else:
            # Already long term, suggest smaller extension
            new_term = min(current_term * 1.5, MAX_LOAN_TERM_MONTHS)
            suggestions.append({
                'new_term': int(new_term),
                'description': f"Extend from {current_term} months to {int(new_term)} months (50% increase)"
            })
    
    # Add additional suggestions based on current term
    if current_term < MAX_LOAN_TERM_MONTHS:
        # Always suggest nearest standard term
        standard_terms = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 144, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, MAX_LOAN_TERM_MONTHS]
        
        for term in standard_terms:
            if term > current_term and term <= MAX_LOAN_TERM_MONTHS:
                # Check if this is a reasonable extension
                if term <= current_term * 2:  # Don't more than double the term
                    suggestions.append({
                        'new_term': term,
                        'description': f"Extend to standard {term//12}-year term ({term} months)"
                    })
                break
    
    return suggestions

def get_interest_rate_increase(risk_score, current_rate):
    """Calculate appropriate interest rate increase based on risk"""
    if risk_score > 85:
        return min(5, 30 - current_rate)  # Max 5% increase, cap at 30%
    elif risk_score > 75:
        return min(3, 30 - current_rate)  # Max 3% increase, cap at 30%
    elif risk_score > 65:
        return min(2, 30 - current_rate)  # Max 2% increase, cap at 30%
    else:
        return min(1, 30 - current_rate)  # Max 1% increase, cap at 30%

def calculate_collateral_needed(loan_amount, risk_score):
    """Calculate collateral amount needed based on risk"""
    if risk_score > 80:
        return loan_amount * 1.5  # 150% collateral for very high risk
    elif risk_score > 70:
        return loan_amount * 1.25  # 125% collateral for high risk
    elif risk_score > 60:
        return loan_amount * 1.1  # 110% collateral for moderate-high risk
    else:
        return loan_amount  # 100% collateral for moderate risk

def calculate_collateral_percentage(risk_score):
    """Calculate collateral percentage based on risk"""
    if risk_score > 80:
        return 150
    elif risk_score > 70:
        return 125
    elif risk_score > 60:
        return 110
    else:
        return 100

def get_disbursement_stages(loan_amount, risk_score):
    """Get staged disbursement plan"""
    if risk_score > 80:
        # Very high risk - 4 stages
        return [
            f"Stage 1: ₹{loan_amount * 0.25:,.2f} (25%) - Initial approval",
            f"Stage 2: ₹{loan_amount * 0.25:,.2f} (25%) - After 3 months of timely payments",
            f"Stage 3: ₹{loan_amount * 0.25:,.2f} (25%) - After 6 months of timely payments",
            f"Stage 4: ₹{loan_amount * 0.25:,.2f} (25%) - After 9 months of timely payments"
        ]
    elif risk_score > 70:
        # High risk - 3 stages
        return [
            f"Stage 1: ₹{loan_amount * 0.40:,.2f} (40%) - Initial approval",
            f"Stage 2: ₹{loan_amount * 0.30:,.2f} (30%) - After 4 months of timely payments",
            f"Stage 3: ₹{loan_amount * 0.30:,.2f} (30%) - After 8 months of timely payments"
        ]
    else:
        # Moderate-high risk - 2 stages
        return [
            f"Stage 1: ₹{loan_amount * 0.60:,.2f} (60%) - Initial approval",
            f"Stage 2: ₹{loan_amount * 0.40:,.2f} (40%) - After 3 months of timely payments"
        ]

@app.route('/dashboard')
def dashboard():
    """Admin dashboard"""
    # Load some statistics from the dataset
    try:
        df = pd.read_csv('data/loan_default.csv')
        
        # Convert Income to Monthly if needed (assuming dataset has annual income)
        if 'Income' in df.columns:
            df['MonthlyIncome'] = df['Income'] / 12
        
        stats = {
            'total_records': len(df),
            'default_count': df['Default'].sum(),
            'default_rate': round((df['Default'].sum() / len(df)) * 100, 2),
            'avg_monthly_income': round(df['MonthlyIncome'].mean(), 2) if 'MonthlyIncome' in df.columns else round(df['Income'].mean() / 12, 2),
            'avg_credit_score': round(df['CreditScore'].mean(), 2),
            'avg_loan_amount': round(df['LoanAmount'].mean(), 2),
            'common_loan_purpose': df['LoanPurpose'].mode()[0] if not df['LoanPurpose'].mode().empty else 'N/A',
            'high_risk_employment': df[df['Default'] == 1]['EmploymentType'].mode()[0] if not df[df['Default'] == 1]['EmploymentType'].mode().empty else 'N/A',
            'max_loan_term': MAX_LOAN_TERM_MONTHS
        }
        
        # Prepare chart data
        default_by_purpose = df.groupby('LoanPurpose')['Default'].mean().to_dict()
        default_by_employment = df.groupby('EmploymentType')['Default'].mean().to_dict()
        
        chart_data = {
            'default_by_purpose': {k: round(v * 100, 2) for k, v in default_by_purpose.items()},
            'default_by_employment': {k: round(v * 100, 2) for k, v in default_by_employment.items()}
        }
        
        return render_template('dashboard.html', stats=stats, chart_data=chart_data)
        
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        # Return default values if file not found
        stats = {
            'total_records': 1000,
            'default_count': 150,
            'default_rate': 15.0,
            'avg_monthly_income': 7083,  # $85,000 annual / 12 months
            'avg_credit_score': 650,
            'avg_loan_amount': 120000,
            'common_loan_purpose': 'Home Loan',
            'high_risk_employment': 'Unemployed',
            'max_loan_term': MAX_LOAN_TERM_MONTHS
        }
        
        chart_data = {
            'default_by_purpose': {'Auto': 12, 'Business': 18, 'Education': 8, 'Home': 5, 'Other': 15},
            'default_by_employment': {'Full-time': 5, 'Part-time': 12, 'Self-employed': 15, 'Unemployed': 25}
        }
        
        return render_template('dashboard.html', stats=stats, chart_data=chart_data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate loan term
        loan_term = int(data.get('LoanTerm', 0))
        if loan_term > MAX_LOAN_TERM_MONTHS:
            return jsonify({'error': f'Loan term cannot exceed {MAX_LOAN_TERM_MONTHS} months'}), 400
        
        # Preprocess and predict
        scaled_features, raw_features = preprocess_input(data)
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        # Determine risk level
        risk_score = probability[1] * 100
        if risk_score < 20:
            risk_level = 'Low'
        elif risk_score < 50:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Generate manager suggestions for high-risk
        manager_suggestions = []
        if risk_score > 50:
            manager_suggestions = generate_manager_suggestions(data, raw_features, risk_score)
        
        result = {
            'prediction': int(prediction),
            'default_probability': float(probability[1]),
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'monthly_income': float(raw_features.get('MonthlyIncome', 0)),
            'annual_income': float(raw_features.get('AnnualIncome', 0)),
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'recommendations': generate_recommendations(data, risk_score, raw_features),
            'manager_suggestions': manager_suggestions,
            'is_high_risk': risk_score > 50,
            'max_loan_term': MAX_LOAN_TERM_MONTHS
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'features_used': len(feature_names) if feature_names else 0,
        'max_loan_term_months': MAX_LOAN_TERM_MONTHS
    })

def generate_recommendations(data, risk_score, raw_features=None):
    """Generate recommendations based on risk factors"""
    recommendations = []
    
    # Get customer name for personalization
    customer_name = data.get('CustomerName', 'Customer')
    loan_number = data.get('LoanNumber', 'N/A')
    
    # Get income values
    monthly_income = float(data.get('MonthlyIncome', 0))
    annual_income = monthly_income * 12
    
    if raw_features:
        monthly_income = raw_features.get('MonthlyIncome', monthly_income)
        annual_income = raw_features.get('AnnualIncome', annual_income)
    
    # Loan identification
    recommendations.append(f"💰 Monthly Income: ₹{monthly_income:,.2f} (Annual: ₹{annual_income:,.2f})")
    
    # Credit Score recommendations
    credit_score = int(data.get('CreditScore', 0))
    if credit_score < 600:
        recommendations.append("⚠️ Low credit score: Consider improving credit history before applying")
    elif credit_score < 700:
        recommendations.append("⚠️ Moderate credit score: Provide additional income documentation")
    elif credit_score >= 750:
        recommendations.append("✅ Excellent credit score: Favorable factor")
    
    # DTI recommendations
    dti = float(data.get('DTIRatio', 0))
    if dti > 0.5:
        recommendations.append(f"⚠️ High DTI ratio ({dti:.2%}): Consider reducing existing debt")
    elif dti > 0.4:
        recommendations.append(f"⚠️ Moderate DTI ratio ({dti:.2%}): Maintain current debt levels")
    else:
        recommendations.append(f"✅ Good DTI ratio ({dti:.2%}): Favorable factor")
    
    # Employment recommendations
    employment = data.get('EmploymentType', '')
    if employment == 'Unemployed':
        recommendations.append("⚠️ Unemployed status: Strongly recommend a co-signer")
    elif employment == 'Self-employed':
        recommendations.append("⚠️ Self-employed: Provide 2 years of tax returns")
    elif employment == 'Part-time':
        recommendations.append("⚠️ Part-time employment: Consider adding a secondary income source")
    elif employment == 'Full-time':
        recommendations.append("✅ Full-time employment: Favorable factor")
    
    # Income adequacy check
    loan_amount = float(data.get('LoanAmount', 0))
    monthly_payment = calculate_monthly_payment(loan_amount, 
                                               float(data.get('InterestRate', 0)), 
                                               int(data.get('LoanTerm', 0)))
    
    if monthly_income > 0:
        payment_to_income = monthly_payment / monthly_income
        
        if payment_to_income > 0.35:
            recommendations.append(f"⚠️ High payment-to-income ratio ({payment_to_income:.1%}): Monthly payment is {payment_to_income:.1%} of income")
        elif payment_to_income > 0.25:
            recommendations.append(f"⚠️ Moderate payment-to-income ratio ({payment_to_income:.1%}): Manageable but monitor")
        else:
            recommendations.append(f"✅ Good payment-to-income ratio ({payment_to_income:.1%}): Favorable factor")
    
    # Interest rate recommendations
    interest_rate = float(data.get('InterestRate', 0))
    if interest_rate > 15:
        recommendations.append(f"⚠️ High interest rate ({interest_rate}%): Shop for better rates")
    elif interest_rate > 10:
        recommendations.append(f"⚠️ Moderate interest rate ({interest_rate}%): Standard market rate")
    else:
        recommendations.append(f"✅ Competitive interest rate ({interest_rate}%): Favorable factor")
    
    # Co-signer recommendations
    has_co_signer = data.get('HasCoSigner', 'No')
    if has_co_signer == 'Yes':
        recommendations.append("✅ Has co-signer: Risk mitigation factor")
    else:
        recommendations.append("⚠️ No co-signer: Consider requiring a co-signer")
    
    # Employment duration
    months_employed = int(data.get('MonthsEmployed', 0))
    if months_employed < 12:
        recommendations.append(f"⚠️ Short employment duration ({months_employed} months): Limited employment history")
    elif months_employed >= 24:
        recommendations.append(f"✅ Stable employment ({months_employed} months): Favorable factor")
    
    # Mortgage status
    has_mortgage = data.get('HasMortgage', 'No')
    if has_mortgage == 'Yes':
        recommendations.append("⚠️ Has existing mortgage: Additional debt obligation")
    
    # Dependents
    has_dependents = data.get('HasDependents', 'No')
    if has_dependents == 'Yes':
        recommendations.append("⚠️ Has dependents: Additional financial obligations")
    
    # Loan term validation
    loan_term = int(data.get('LoanTerm', 0))
    if loan_term > 240:  # More than 20 years
        recommendations.append(f"⚠️ Long loan term ({loan_term} months): Consider shorter term for faster equity build-up")
    
    # General recommendation based on risk score
    if risk_score > 70:
        recommendations.append("❌ VERY HIGH RISK: Strongly recommend loan rejection")
    elif risk_score > 50:
        recommendations.append("⚠️ HIGH RISK: Require additional collateral or co-signer")
    elif risk_score > 30:
        recommendations.append("⚠️ MEDIUM RISK: Standard verification procedures recommended")
    else:
        recommendations.append("✅ LOW RISK: Expedited processing available")
    
    return recommendations

def calculate_monthly_payment(loan_amount, annual_interest_rate, loan_term_months):
    """Calculate monthly loan payment"""
    if loan_term_months == 0:
        return 0
    
    monthly_rate = annual_interest_rate / 100 / 12
    if monthly_rate == 0:
        return loan_amount / loan_term_months
    
    payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** loan_term_months) / \
              ((1 + monthly_rate) ** loan_term_months - 1)
    return payment

@app.route('/api/features/importance')
def feature_importance():
    """Endpoint to get feature importance"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get feature importance if available (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance_dict = dict(zip(feature_names, importance.tolist()))
            
            # Sort by importance
            sorted_importance = sorted(feature_importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True)
            
            return jsonify({
                'feature_importance': dict(sorted_importance[:10]),  # Top 10 features
                'total_features': len(feature_names),
                'status': 'success'
            })
        else:
            return jsonify({
                'message': 'Feature importance not available for this model type',
                'features': feature_names,
                'status': 'success'
            })
            
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/calculate/payment', methods=['POST'])
def calculate_payment():
    """Calculate monthly payment endpoint"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        loan_amount = float(data.get('loan_amount', 0))
        interest_rate = float(data.get('interest_rate', 0))
        loan_term = int(data.get('loan_term', 0))
        
        # Validate loan term
        if loan_term > MAX_LOAN_TERM_MONTHS:
            return jsonify({
                'error': f'Loan term cannot exceed {MAX_LOAN_TERM_MONTHS} months',
                'max_allowed_term': MAX_LOAN_TERM_MONTHS
            }), 400
        
        monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_term)
        
        return jsonify({
            'monthly_payment': round(monthly_payment, 2),
            'total_payment': round(monthly_payment * loan_term, 2),
            'total_interest': round(monthly_payment * loan_term - loan_amount, 2),
            'status': 'success',
            'max_allowed_term': MAX_LOAN_TERM_MONTHS
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/manager/suggestions', methods=['POST'])
def manager_suggestions():
    """API endpoint for manager suggestions"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate loan term
        loan_term = int(data.get('LoanTerm', 0))
        if loan_term > MAX_LOAN_TERM_MONTHS:
            return jsonify({'error': f'Loan term cannot exceed {MAX_LOAN_TERM_MONTHS} months'}), 400
        
        # Get risk score from request or calculate it
        risk_score = float(data.get('risk_score', 0))
        
        # If no risk score provided, calculate it
        if risk_score == 0:
            scaled_features, raw_features = preprocess_input(data)
            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0]
            risk_score = probability[1] * 100
        
        # Generate manager suggestions
        suggestions = generate_manager_suggestions(data, {}, risk_score)
        
        return jsonify({
            'suggestions': suggestions,
            'risk_score': risk_score,
            'is_high_risk': risk_score > 50,
            'max_loan_term': MAX_LOAN_TERM_MONTHS,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/tenure/suggestions', methods=['POST'])
def tenure_suggestions():
    """API endpoint specifically for tenure suggestions"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        current_term = int(data.get('current_term', 0))
        risk_score = float(data.get('risk_score', 50))
        loan_amount = float(data.get('loan_amount', 0))
        interest_rate = float(data.get('interest_rate', 0))
        monthly_income = float(data.get('monthly_income', 0))
        
        # Validate current term
        if current_term > MAX_LOAN_TERM_MONTHS:
            return jsonify({
                'error': f'Current term cannot exceed {MAX_LOAN_TERM_MONTHS} months',
                'max_allowed_term': MAX_LOAN_TERM_MONTHS
            }), 400
        
        # Get tenure suggestions
        suggestions = []
        tenure_options = get_tenure_suggestions(current_term, risk_score)
        
        for option in tenure_options:
            new_term = option['new_term']
            current_payment = calculate_monthly_payment(loan_amount, interest_rate, current_term)
            new_payment = calculate_monthly_payment(loan_amount, interest_rate, new_term)
            payment_reduction = current_payment - new_payment
            reduction_percentage = (payment_reduction / current_payment) * 100 if current_payment > 0 else 0
            payment_to_income = (new_payment / monthly_income) * 100 if monthly_income > 0 else 0
            
            suggestions.append({
                'current_term': current_term,
                'suggested_term': new_term,
                'term_increase_months': new_term - current_term,
                'term_increase_percentage': ((new_term - current_term) / current_term) * 100 if current_term > 0 else 0,
                'current_payment': round(current_payment, 2),
                'suggested_payment': round(new_payment, 2),
                'payment_reduction': round(payment_reduction, 2),
                'payment_reduction_percentage': round(reduction_percentage, 2),
                'payment_to_income_ratio': round(payment_to_income, 2),
                'description': option['description'],
                'is_recommended': payment_to_income <= 35 if monthly_income > 0 else True
            })
        
        return jsonify({
            'suggestions': suggestions,
            'current_term': current_term,
            'risk_score': risk_score,
            'max_allowed_term': MAX_LOAN_TERM_MONTHS,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    # Try to initialize the app
    if init_app():
        print("✅ Application initialized successfully")
        print(f"✅ Model loaded with {len(feature_names) if feature_names else 0} features")
        print(f"✅ Maximum loan term: {MAX_LOAN_TERM_MONTHS} months ({MAX_LOAN_TERM_MONTHS//12} years)")
        print("✅ Monthly income input enabled (converted to annual for model)")
        print("✅ Manager suggestions module activated with parameter-based tenure logic")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize application. Check model files.")
        print("Running in limited mode...")
        app.run(debug=True, host='0.0.0.0', port=5000)