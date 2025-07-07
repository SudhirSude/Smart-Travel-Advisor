import os
import PyPDF2
import numpy as np
import faiss
import json
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import joblib


class TravelAssistant:
    def __init__(self):
        # Initialize components safely
        load_dotenv()
        
        # Initialize embedding model first
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Single Groq client instance
        try:
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        except Exception as e:
            print(f"Error initializing Groq client: {str(e)}")
            self.client = None
        
        # Initialize RAG components
        self.index = None
        self.documents = []
        
        # Initialize ML model
        try:
            self.price_model = joblib.load('app/models/flight_price_model.joblib')
            self.model_metadata = joblib.load('app/models/model_metadata.joblib')
        except Exception as e:
            print(f"Error loading ML model: {str(e)}")
            self.price_model = None
        
        # Initialize RAG system
        self._init_rag_system()


    def _init_rag_system(self):
        """Initialize RAG with travel documents"""
        try:
            # Initialize embedding model if not already done
            if not hasattr(self, 'embedding_model'):
                self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
            
            pdf_dir = "app/data/travel_docs"
            if os.path.exists(pdf_dir) and os.listdir(pdf_dir):
                print(f"Found {len(os.listdir(pdf_dir))} PDFs in {pdf_dir}")
                self.load_and_chunk_pdfs(pdf_dir)
                self.create_faiss_index()
                print("RAG system initialized successfully")
            else:
                print(f"Warning: No PDFs found in {pdf_dir}")
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            raise    


    
    def load_and_chunk_pdfs(self, pdf_dir="app/data/travel_docs"):
        """Load PDFs and split into chunks (500-800 chars)"""
        self.documents = []  # Reset documents
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                try:
                    with open(os.path.join(pdf_dir, filename), 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                        chunks = [text[i:i+700] for i in range(0, len(text), 700)]
                        self.documents.extend(chunks)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    
    def create_faiss_index(self):
        """Generate embeddings and build FAISS index"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_and_chunk_pdfs() first.")
            
        embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def query_llama3(self, query, context, max_tokens=200):
        """Query Llama 3 via Groq API with RAG context"""
        prompt = f"""As a travel expert, answer using ONLY the provided context.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer concisely in 2-3 sentences. If unsure, say "I couldn't find definitive information.":"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Less creative but more factual
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def search_documents(self, query, k=3):
        """Retrieve top-k relevant document chunks"""
        if not self.index:
            raise ValueError("FAISS index not initialized.")
            
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return [self.documents[i] for i in indices[0]]
    
    def answer_question(self, query):
        """End-to-end RAG pipeline"""
        try:
            relevant_chunks = self.search_documents(query)
            context = "\n\n---\n\n".join(relevant_chunks)
            return self.query_llama3(query, context)
        except Exception as e:
            return f"System error: {str(e)}"
        
    def detect_flight_query(self, query):
        """Check if query is about flight prices"""
        price_keywords = ['price', 'cost', 'fare', 'how much', 'predict']
        route_keywords = ['from', 'to', 'flight']
        return any(kw in query.lower() for kw in price_keywords) and \
               any(kw in query.lower() for kw in route_keywords)
    
    def extract_flight_params(self, query):
        """Extract flight parameters from natural language query"""
        try:
            # Use LLM to extract structured data
            prompt = f"""Extract flight details from this query as JSON:
            Query: "{query}"
            
            Return ONLY JSON with these keys:
            {{
                "airline": "defaults to Indigo if not specified",
                "source": "",
                "destination": "",
                "date": "today if not specified (format: DD/MM/YYYY)",
                "dep_time": "morning/afternoon/evening/night or exact time",
                "stops": "non-stop/1 stop/2 stops/etc"
            }}"""
            
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            params = json.loads(response.choices[0].message.content)
            
            # Convert to model input format
            return self._prepare_model_input(params)
            
        except Exception as e:
            print(f"Error extracting flight params: {str(e)}")
            return None
        
    def _prepare_model_input(self, params):
        """Convert extracted params to model input format"""
        # Parse date
        date_str = params.get('date', 'today')
        if date_str.lower() == 'today':
            journey_date = datetime.now()
        else:
            journey_date = datetime.strptime(date_str, '%d/%m/%Y')
        
        # Parse departure time
        dep_time = params.get('dep_time', 'morning')
        if dep_time.isdigit():  # Exact time like "14:30"
            dep_hour = int(dep_time.split(':')[0])
            dep_min = int(dep_time.split(':')[1])
        else:  # Time of day
            if 'morning' in dep_time.lower():
                dep_hour, dep_min = 9, 0
            elif 'afternoon' in dep_time.lower():
                dep_hour, dep_min = 14, 0
            elif 'evening' in dep_time.lower():
                dep_hour, dep_min = 18, 0
            else:  # night
                dep_hour, dep_min = 21, 0
        
        # Estimate duration based on route (simplified)
        common_routes = {
            ('Banglore', 'New Delhi'): 170,
            ('Delhi', 'Cochin'): 210,
            ('Kolkata', 'Banglore'): 150
        }
        duration = common_routes.get(
            (params['source'], params['destination']), 120
        )
        
        return {
            'Airline': params.get('airline', 'Indigo'),
            'Source': params['source'],
            'Destination': params['destination'],
            'Route': f"{params['source']} → {params['destination']}",
            'Total_Stops': params.get('stops', 'non-stop'),
            'Journey_Day': journey_date.day,
            'Journey_Month': journey_date.month,
            'Journey_Year': journey_date.year,
            'Duration_Mins': duration,
            'Dep_Hour': dep_hour,
            'Dep_Min': dep_min,
            'Arrival_Hour': (dep_hour + (duration // 60)) % 24,
            'Arrival_Min': dep_min + (duration % 60),
            'Meal_Included': 0
        }
    
    def predict_flight_price(self, input_features):
        """Make price prediction using the ML model"""
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([input_features])
            
            # Ensure all columns are present
            for col in self.model_metadata['features']:
                if col not in input_df.columns:
                    input_df[col] = np.nan
            
            # Reorder columns
            input_df = input_df[self.model_metadata['features']]
            
            # Predict
            prediction = self.price_model.predict(input_df)
            return round(float(prediction[0]), None), None
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
        
    def process_query(self, query):
        """Main method to handle all queries"""
        if self.detect_flight_query(query):
            # Flight price prediction
            params = self.extract_flight_params(query)
            if not params:
                return "Sorry, I couldn't understand the flight details. Please try again with details like: 'Predict flight price from Delhi to Mumbai on July 20 with Indigo'"
            
            price, error = self.predict_flight_price(params)
            if error:
                return f"⚠️ {error}"
            
            # Format response
            airline = params['Airline']
            source = params['Source']
            dest = params['Destination']
            date = f"{params['Journey_Day']}/{params['Journey_Month']}/{params['Journey_Year']}"
            stops = params['Total_Stops']
            
            return f"""✈️ Flight Price Prediction:
- **Route**: {source} to {dest}
- **Airline**: {airline}
- **Date**: {date}
- **Stops**: {stops}
- **Estimated Price**: ₹{price:,.2f}

*Note: This is an estimate based on historical data. Actual prices may vary.*"""
        else:
            # Handle as RAG query
            if not self.index:
                return "⚠️ Travel knowledge base is still loading. Please try again later."
            
            try:
                relevant_chunks = self.search_documents(query)
                context = "\n\n---\n\n".join(relevant_chunks)
                return self.query_llama3(query, context)
            except Exception as e:
                return f"❌ Error: {str(e)}"
