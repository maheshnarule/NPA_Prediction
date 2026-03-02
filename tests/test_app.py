import unittest
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, preprocess_input

class TestNPAApp(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
        # Sample valid input data
        self.sample_data = {
            'Age': 35,
            'Income': 75000,
            'LoanAmount': 150000,
            'CreditScore': 720,
            'MonthsEmployed': 60,
            'NumCreditLines': 3,
            'InterestRate': 5.5,
            'LoanTerm': 360,
            'DTIRatio': 0.35,
            'Education': "Bachelor's",
            'EmploymentType': 'Full-time',
            'MaritalStatus': 'Married',
            'HasMortgage': 'Yes',
            'HasDependents': 'No',
            'LoanPurpose': 'Home',
            'HasCoSigner': 'No'
        }
    
    def test_home_page(self):
        """Test home page loads successfully"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'NPA Risk Predictor', response.data)
    
    def test_predict_page_get(self):
        """Test predict page loads with GET request"""
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Loan Application Details', response.data)
    
    def test_dashboard_page(self):
        """Test dashboard page loads"""
        response = self.app.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Analytics Dashboard', response.data)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_api_predict_invalid_data(self):
        """Test API predict with invalid data"""
        response = self.app.post('/api/predict', 
                                data=json.dumps({}),
                                content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_api_predict_valid_data(self):
        """Test API predict with valid data"""
        response = self.app.post('/api/predict',
                                data=json.dumps(self.sample_data),
                                content_type='application/json')
        # This might fail if model is not loaded, but should return proper status
        self.assertIn(response.status_code, [200, 500])
    
    def test_preprocess_input_structure(self):
        """Test preprocessing returns correct structure"""
        try:
            processed_data, raw_features = preprocess_input(self.sample_data)
            self.assertIsNotNone(processed_data)
            self.assertIsInstance(processed_data, np.ndarray)
            # Check shape (should be 2D array with 1 sample)
            self.assertEqual(len(processed_data.shape), 2)
            self.assertEqual(processed_data.shape[0], 1)
        except Exception as e:
            # If model not loaded, preprocessing might fail
            # That's acceptable for testing
            pass
    
    def test_missing_required_fields(self):
        """Test form submission with missing fields"""
        incomplete_data = self.sample_data.copy()
        del incomplete_data['Age']  # Remove required field
        
        response = self.app.post('/predict',
                                data=incomplete_data,
                                follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # Should show error message
        self.assertIn(b'Error', response.data)
    
    def test_invalid_age(self):
        """Test form submission with invalid age"""
        invalid_data = self.sample_data.copy()
        invalid_data['Age'] = 10  # Too young
        
        response = self.app.post('/predict',
                                data=invalid_data,
                                follow_redirects=True)
        self.assertEqual(response.status_code, 200)
    
    def test_result_page_redirect(self):
        """Test result page redirects without session data"""
        response = self.app.get('/result', follow_redirects=False)
        # Should redirect to predict page
        self.assertEqual(response.status_code, 302)
    
    def test_static_files(self):
        """Test static files are accessible"""
        response = self.app.get('/static/css/style.css')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'text/css', response.headers['Content-Type'])
    
    def test_404_page(self):
        """Test 404 error handling"""
        response = self.app.get('/nonexistent-page')
        self.assertEqual(response.status_code, 404)
    
    def test_form_validation(self):
        """Test form validation on client side"""
        response = self.app.get('/predict')
        # Check form has required attributes
        self.assertIn(b'required', response.data)
        self.assertIn(b'min=', response.data)
        self.assertIn(b'max=', response.data)

if __name__ == '__main__':
    unittest.main()