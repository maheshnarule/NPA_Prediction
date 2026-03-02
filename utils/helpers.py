import pandas as pd
import numpy as np
from datetime import datetime
import json
import re

def validate_input_data(data):
    """
    Validate input data for prediction
    """
    errors = []
    
    # Age validation
    if 'Age' in data:
        age = int(data['Age'])
        if age < 18 or age > 100:
            errors.append("Age must be between 18 and 100")
    
    # Income validation
    if 'Income' in data:
        income = float(data['Income'])
        if income < 0:
            errors.append("Income cannot be negative")
    
    # Credit score validation
    if 'CreditScore' in data:
        score = int(data['CreditScore'])
        if score < 300 or score > 850:
            errors.append("Credit score must be between 300 and 850")
    
    # DTI validation
    if 'DTIRatio' in data:
        dti = float(data['DTIRatio'])
        if dti < 0 or dti > 1:
            errors.append("DTI ratio must be between 0 and 1")
    
    # Interest rate validation
    if 'InterestRate' in data:
        rate = float(data['InterestRate'])
        if rate < 0 or rate > 30:
            errors.append("Interest rate must be between 0 and 30%")
    
    return errors

def format_currency(value):
    """
    Format number as currency
    """
    try:
        value = float(value)
        return f"${value:,.0f}"
    except:
        return str(value)

def calculate_risk_factors(data):
    """
    Calculate additional risk factors from input data
    """
    risk_factors = []
    
    # Credit score risk
    credit_score = int(data.get('CreditScore', 0))
    if credit_score < 600:
        risk_factors.append({
            'factor': 'Low Credit Score',
            'score': credit_score,
            'risk': 'High',
            'description': f'Credit score of {credit_score} is below acceptable threshold'
        })
    
    # DTI risk
    dti = float(data.get('DTIRatio', 0))
    if dti > 0.5:
        risk_factors.append({
            'factor': 'High DTI Ratio',
            'score': dti,
            'risk': 'High',
            'description': f'DTI ratio of {dti:.2f} exceeds recommended limit of 0.5'
        })
    
    # Employment risk
    employment = data.get('EmploymentType', '')
    if employment == 'Unemployed':
        risk_factors.append({
            'factor': 'Unemployed',
            'score': 'N/A',
            'risk': 'High',
            'description': 'Applicant is currently unemployed'
        })
    
    # Loan-to-income ratio
    income = float(data.get('Income', 0))
    loan_amount = float(data.get('LoanAmount', 0))
    if income > 0:
        lti_ratio = loan_amount / income
        if lti_ratio > 5:
            risk_factors.append({
                'factor': 'High Loan-to-Income',
                'score': f'{lti_ratio:.1f}x',
                'risk': 'Medium',
                'description': f'Loan amount is {lti_ratio:.1f} times annual income'
            })
    
    return risk_factors

def generate_report(result, input_data, risk_factors=None):
    """
    Generate a comprehensive report
    """
    report = {
        'report_id': f"NPA-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'prediction': result['prediction'],
            'risk_score': result['risk_score'],
            'risk_level': result['risk_level'],
            'default_probability': result['default_probability']
        },
        'applicant_info': {
            'age': input_data.get('Age'),
            'education': input_data.get('Education'),
            'employment': input_data.get('EmploymentType'),
            'marital_status': input_data.get('MaritalStatus')
        },
        'financial_info': {
            'income': format_currency(input_data.get('Income')),
            'credit_score': input_data.get('CreditScore'),
            'dti_ratio': input_data.get('DTIRatio'),
            'num_credit_lines': input_data.get('NumCreditLines')
        },
        'loan_info': {
            'amount': format_currency(input_data.get('LoanAmount')),
            'purpose': input_data.get('LoanPurpose'),
            'term': input_data.get('LoanTerm'),
            'interest_rate': f"{input_data.get('InterestRate')}%"
        },
        'recommendations': result.get('specific_recommendations', []),
        'risk_factors': risk_factors if risk_factors else []
    }
    
    return report

def save_prediction_to_history(prediction_data):
    """
    Save prediction to history file
    """
    try:
        # Load existing history
        try:
            with open('data/prediction_history.json', 'r') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        # Add new prediction
        history.append(prediction_data)
        
        # Keep only last 100 predictions
        if len(history) > 100:
            history = history[-100:]
        
        # Save back
        with open('data/prediction_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving prediction history: {str(e)}")
        return False

def load_prediction_history(limit=50):
    """
    Load prediction history
    """
    try:
        with open('data/prediction_history.json', 'r') as f:
            history = json.load(f)
        return history[-limit:] if limit else history
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading prediction history: {str(e)}")
        return []

def calculate_portfolio_stats():
    """
    Calculate portfolio statistics
    """
    try:
        df = pd.read_csv('data/loan_default.csv')
        
        stats = {
            'total_loans': len(df),
            'default_count': int(df['Default'].sum()),
            'default_rate': float(df['Default'].mean() * 100),
            'avg_income': float(df['Income'].mean()),
            'avg_credit_score': float(df['CreditScore'].mean()),
            'avg_loan_amount': float(df['LoanAmount'].mean()),
            'avg_interest_rate': float(df['InterestRate'].mean()),
            'most_common_purpose': df['LoanPurpose'].mode().iloc[0] if not df['LoanPurpose'].mode().empty else 'N/A',
            'highest_default_purpose': df.groupby('LoanPurpose')['Default'].mean().idxmax() if not df.groupby('LoanPurpose')['Default'].mean().empty else 'N/A'
        }
        
        return stats
    except Exception as e:
        print(f"Error calculating portfolio stats: {str(e)}")
        return {}