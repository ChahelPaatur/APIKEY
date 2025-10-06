# ExoML Deployment Guide

This guide covers both local development and Vercel deployment.

## Local Development

### Quick Start
```bash
# Make the script executable (Mac/Linux)
chmod +x start_server.sh
./start_server.sh

# Or on Windows
start_server.bat
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask server
python app.py
```

Then open `exo.html` in your browser and upload a CSV file.

---

## Vercel Deployment

### Prerequisites
1. Install Vercel CLI: `npm install -g vercel`
2. Have a Vercel account (free tier works!)

### File Structure for Vercel
```
HTML-for-ExoML/
├── index.html              # Frontend web app
├── vercel.json             # Vercel configuration
├── requirements.txt        # Python dependencies
├── .vercelignore          # Files to ignore
├── api/
│   ├── predict.py         # Prediction API endpoint
│   └── health.py          # Health check endpoint
└── model_files/
    ├── exoplanet_bilstm.h5 # BiLSTM model (5.1MB)
    ├── scaler.pkl          # Data scaler
    └── metadata.pkl        # Model metadata
```

### Deployment Steps

#### Option 1: Deploy via Vercel CLI (Recommended)

1. **Login to Vercel**
   ```bash
   vercel login
   ```

2. **Deploy from project directory**
   ```bash
   cd HTML-for-ExoML
   vercel
   ```

3. **Follow the prompts:**
   - Set up and deploy? **Y**
   - Which scope? Select your account
   - Link to existing project? **N** (first time)
   - What's your project's name? **exoml** (or your choice)
   - In which directory is your code located? **./`**
   - Want to override the settings? **N**

4. **Deploy to production**
   ```bash
   vercel --prod
   ```

#### Option 2: Deploy via Vercel Dashboard

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your Git repository (GitHub, GitLab, or Bitbucket)
3. Vercel will auto-detect the configuration from `vercel.json`
4. Click "Deploy"

### Important Notes

#### Model File Size
- Vercel has a 250MB limit for serverless functions
- If your model is too large, consider:
  - Using model quantization to reduce size
  - Storing the model in cloud storage (S3, GCS) and loading it at runtime
  - Using a lighter model architecture

#### Cold Starts
- First request may take 10-30 seconds (cold start)
- Subsequent requests will be faster
- Consider upgrading to Vercel Pro for better performance

#### Memory Configuration
The `vercel.json` is configured for maximum memory (3008MB):
```json
"functions": {
  "api/*.py": {
    "memory": 3008,
    "maxDuration": 60
  }
}
```

### Testing Your Deployment

After deployment, test your endpoints:

1. **Health Check**
   ```bash
   curl https://your-app.vercel.app/api/health
   ```

2. **Upload CSV through the website**
   - Visit `https://your-app.vercel.app`
   - Sign in (Firebase auth still works)
   - Upload a CSV file
   - The app auto-detects it's on Vercel and uses `/api/predict`

### Environment Variables

If you need to add environment variables:

1. **Via Vercel CLI:**
   ```bash
   vercel env add VARIABLE_NAME
   ```

2. **Via Dashboard:**
   - Go to your project settings
   - Navigate to "Environment Variables"
   - Add your variables

### Troubleshooting

#### Model Loading Issues
- Check Vercel function logs: `vercel logs`
- Ensure model files are not in `.vercelignore`
- Verify file paths in `api/predict.py`

#### CORS Errors
- The API endpoints include CORS headers
- If issues persist, check browser console for details

#### Timeout Errors
- Default timeout is 60s (max on Hobby plan)
- Optimize model inference or upgrade to Pro plan

#### Memory Errors
- Reduce model size
- Use model quantization
- Consider splitting predictions into batches

### Updating Your Deployment

```bash
# Make changes to your code
git add .
git commit -m "Update model or frontend"
git push

# Or deploy directly with Vercel CLI
vercel --prod
```

### Custom Domain

1. Go to your project in Vercel dashboard
2. Settings → Domains
3. Add your custom domain
4. Follow DNS configuration instructions

---

## API Documentation

### POST /api/predict

**Request:**
```json
{
  "data": "FLUX.1,FLUX.2,FLUX.3...\n93.85,83.81,20.10..."
}
```

**Response:**
```json
{
  "isExoplanet": true,
  "confidence": 85.5,
  "probability": 0.855,
  "prediction": 1,
  "orbitalPeriod": 5.3,
  "temperature": 1200.5,
  "transitDepth": 0.001,
  "planetType": "Hot Jupiter",
  "modelType": "BiLSTM Neural Network",
  "numSamples": 1
}
```

### GET /api/health

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "modelType": "BiLSTM"
}
```

---

## Cost Considerations

### Vercel Free (Hobby) Tier:
- ✅ Unlimited deployments
- ✅ 100GB bandwidth
- ✅ Serverless function execution
- ⚠️ 10 second timeout (60s with config)
- ⚠️ Cold starts
- ⚠️ No commercial use

### Vercel Pro ($20/month):
- ✅ Everything in Free
- ✅ Faster cold starts
- ✅ Commercial use allowed
- ✅ Priority support

---

## Alternative Deployment Options

If Vercel doesn't work for your model size:

1. **Railway.app** - Better for larger models
2. **Render.com** - Free tier with Docker support
3. **Heroku** - Classic PaaS option
4. **AWS Lambda + API Gateway** - More complex but scalable
5. **Google Cloud Run** - Container-based, good for ML

Need help? Check Vercel's [Python deployment docs](https://vercel.com/docs/functions/serverless-functions/runtimes/python).

