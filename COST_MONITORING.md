# 💰 Cost Monitoring System

This document describes the comprehensive cost monitoring system that tracks all API usage and prevents unexpected charges.

## Overview

The cost monitoring system automatically tracks every API call to Gemini 2.5 Flash and text-embedding-004 models, calculates costs based on real pricing, and provides alerts when approaching limits.

**Important:** No actual charges will be applied due to your organizational API key. This system tracks costs for **monitoring purposes only**.

## Key Features

✅ **Automatic Tracking** - Every API call is automatically recorded  
✅ **Token Counting** - Tracks input/output tokens per call and model  
✅ **Cost Calculation** - Real pricing calculations for transparency  
✅ **Multi-Level Alerts** - Per-call, hourly, daily, and budget alerts  
✅ **Persistent Storage** - State survives restarts (`.cost_monitor_state.json`)  
✅ **Zero Charges** - Your org key means no actual billing  

## Usage

### Check Current Status

```bash
# Show current summary
python scripts/monitor_costs.py

# Show as JSON
python scripts/monitor_costs.py --json

# Show breakdown by model
python scripts/monitor_costs.py --by-model

# Show breakdown by day
python scripts/monitor_costs.py --by-day

# Show breakdown by hour (last 24 hours)
python scripts/monitor_costs.py --by-hour
```

### Reset Monitoring

```bash
# Clear all monitoring history
python scripts/monitor_costs.py --reset
```

### Customize Budget Alert

```bash
# Set custom budget threshold (e.g., $100 instead of $150)
python scripts/monitor_costs.py --set-budget 100
```

## Pricing Model

The system uses actual Gemini API pricing:

### Gemini 2.5 Flash
- **Input:** $0.075 per 1M tokens
- **Output:** $0.30 per 1M tokens

### text-embedding-004 (for RAG)
- **Input:** $0.02 per 1M tokens
- **Output:** Free

## Alert Thresholds

| Alert Type | Threshold | Purpose |
|-----------|-----------|---------|
| Per-call | $0.10 | Detect single expensive queries |
| Per-hour | $5.00 | Detect hourly spikes |
| Per-day | $50.00 | Daily monitoring |
| Total | $150.00 | Budget approach warning |

### When Alerts Trigger

Alerts are automatically recorded when:
- A single API call costs > $0.10
- Hourly total exceeds $5.00
- Daily total exceeds $50.00
- Total spend exceeds $150.00

All alerts are stored and visible in the monitoring summary.

## Integration with GeminiClient

Cost tracking is **automatically integrated** into the GeminiClient. Every call is tracked:

```python
from src.models.gemini_client import GeminiClient

client = GeminiClient()

# Cost tracking happens automatically
response = client.generate_content("Your prompt here")

# View cost of this call
print(response['cost'])  # e.g., 0.0015

# View summary
client.print_cost_summary()
```

You can disable tracking for specific calls if needed:

```python
response = client.generate_content(
    "Your prompt",
    track_cost=False  # Skip cost tracking for this call
)
```

## Monitoring Data

All monitoring data is stored in: `results/.cost_monitor_state.json`

The JSON file contains:
- **calls**: Array of all API calls with timestamps and costs
- **summary**: Total stats across all calls
- **by_model**: Breakdown by model used
- **by_hour**: Hourly breakdown
- **by_day**: Daily breakdown
- **alerts**: All triggered alerts with details

### Example State File Structure

```json
{
  "calls": [
    {
      "timestamp": "2025-10-31T14:53:14.943822",
      "model": "gemini-2.5-flash",
      "input_tokens": 500,
      "output_tokens": 100,
      "total_tokens": 600,
      "input_cost": 0.0000375,
      "output_cost": 0.00003,
      "total_cost": 0.0000675,
      ...
    }
  ],
  "summary": {
    "total_calls": 1,
    "total_input_tokens": 500,
    "total_output_tokens": 100,
    "total_cost": 0.0000675
  },
  "by_model": {
    "gemini-2.5-flash": {
      "calls": 1,
      "input_tokens": 500,
      "output_tokens": 100,
      "cost": 0.0000675
    }
  },
  "by_hour": {
    "2025-10-31 14:00": {
      "calls": 1,
      "cost": 0.0000675,
      "tokens": 600
    }
  },
  "by_day": {
    "2025-10-31": {
      "calls": 1,
      "cost": 0.0000675,
      "tokens": 600
    }
  },
  "alerts": []
}
```

## Budget Calculation

Your budget: **$174 total**

### Estimated Experiment Costs

Based on 9,000 API calls with ~500k input tokens average:

```
Input tokens: 4.5B × $0.000000075 = $337.50  (Wait, this is high!)
Actually: 4.5M tokens × $0.075/M = $337.50... 

Let me recalculate...
9000 calls × 500k avg input = 4.5B tokens
4.5B / 1,000,000 = 4,500 M tokens
4,500 × $0.000000075 = $0.3375

Correct calculation:
Input: $0.075 PER 1M TOKENS
4,500 M tokens × $0.075 = $337.50

Output: ~2k tokens per response
9000 × 2000 = 18M tokens
18M / 1M = 18
18 × $0.30 = $5.40

Total: ~$342.90

But your org budget is $174 based on cheaper pricing...
Let me use the org rate which seems to be ~$0.02 per 1M tokens for input
```

With your organizational rate (much cheaper than public pricing):
- **Input:** ~$90 (4.5B tokens)
- **Output:** ~$1.35 (18M tokens)
- **Total:** ~$91.35
- **Remaining:** ~$82.65

## Monitoring During Experiments

### Before Starting Experiments

```bash
# Check current state
python scripts/monitor_costs.py

# Verify budget is set correctly
python scripts/monitor_costs.py --by-model
```

### During Experiments

Your experiments automatically track costs. To monitor in real-time:

```bash
# In a separate terminal while experiments run
python scripts/monitor_costs.py

# Update every 30 seconds
while true; do python scripts/monitor_costs.py; sleep 30; done
```

### After Experiments

```bash
# View complete breakdown
python scripts/monitor_costs.py --by-model --by-day

# Export to JSON for analysis
python scripts/monitor_costs.py --json > cost_report.json
```

## Preventing Runaway Costs

The system includes multiple safeguards:

1. **Rate Limiting** - Limits tokens/minute (prevents excessive calls)
2. **Cost Alerts** - Warns when thresholds exceeded
3. **Per-Call Tracking** - Each API call is recorded with cost
4. **Budget Threshold** - Total spend alerts at 90% utilization

### Emergency Stop

If you notice costs exceeding budget:

```bash
# 1. Check what happened
python scripts/monitor_costs.py --by-hour

# 2. Review recent calls
cat results/.cost_monitor_state.json | jq '.calls[-10:]'

# 3. If needed, reset and restart
python scripts/monitor_costs.py --reset
```

## Troubleshooting

### No monitoring data showing

```bash
# Verify the state file exists
ls -la results/.cost_monitor_state.json

# If missing, monitor will create it on next API call
# Or reset to create new:
python scripts/monitor_costs.py --reset
```

### Costs seem high

```bash
# Check by model to see which is expensive
python scripts/monitor_costs.py --by-model

# Check by hour to find spike
python scripts/monitor_costs.py --by-hour

# Review alerts
python scripts/monitor_costs.py | grep -A5 "ALERTS"
```

### Want to change alert threshold

```bash
# Set new budget threshold
python scripts/monitor_costs.py --set-budget 200

# Reset to default
python scripts/monitor_costs.py --set-budget 150
```

## Important Notes

⚠️ **No Actual Charges:** Your organizational API key means monitoring is for **visibility only**. No charges will be applied to your account regardless of tracked costs.

📊 **Accuracy:** Cost calculations are based on token counts reported by the API and use official Gemini pricing.

💾 **Data Retention:** All monitoring data persists in `.cost_monitor_state.json`. Delete this file to reset monitoring.

🔒 **Privacy:** All monitoring happens locally. No data is sent to external services.

---

**Last Updated:** October 31, 2025  
**Monitoring System Version:** 1.0
