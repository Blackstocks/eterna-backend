# 🚀 One-Click Railway Deployment

## Deploy in 3 Minutes

### Step 1: Click Deploy Button
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/deploy?referralCode=euphoria)

### Step 2: Configure Redis
1. In Railway dashboard, click **"New"** → **"Database"** → **"Add Redis"**
2. Copy the `REDIS_URL` from Redis settings
3. Go to your app's Variables tab
4. Add: `REDIS_URL` = (paste the Redis URL)

### Step 3: Get Your WebSocket URL
After deployment completes:
- Your WebSocket URL: `wss://[your-app-name].up.railway.app`
- Test endpoint: `wss://[your-app-name].up.railway.app/ws/proportional_payout`

### Alternative: Deploy via GitHub

1. **Create GitHub Repository**
   ```bash
   cd backend/server
   git init
   git add .
   git commit -m "Initial commit"
   gh repo create euphoria-trading-backend --public --push
   ```

2. **Connect to Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub"
   - Select your repository
   - Railway auto-deploys on every push

3. **Add Redis** (same as Step 2 above)

## 🎉 That's It!

Your WebSocket server is now live with:
- ✅ Auto-scaling
- ✅ SSL/TLS encryption
- ✅ Global CDN
- ✅ 99.9% uptime
- ✅ Zero configuration

## Update Flutter App

Replace the Railway URL in `lib/config/app_config.dart`:
```dart
return 'wss://your-actual-app-name.up.railway.app';
```

Then rebuild your iOS app!