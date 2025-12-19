"""
Minimal debug - Test từng bước
"""

import sys
import traceback

print("=" * 60)
print("🚀 STEP-BY-STEP DEBUG")
print("=" * 60)

# STEP 1: Test imports
print("\n1️⃣ Testing imports...")
try:
    import asyncio
    print("   ✅ asyncio")
    import aiohttp
    print("   ✅ aiohttp")
    import json
    print("   ✅ json")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# STEP 2: Test config import
print("\n2️⃣ Testing Config import...")
try:
    from config import Config
    print("   ✅ Config imported")
except Exception as e:
    print(f"   ❌ Config import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# STEP 3: Check config values
print("\n3️⃣ Checking Config values...")
try:
    print(f"   📍 VNPT_API_URL: {Config.VNPT_API_URL}")
    print(f"   🔑 ACCESS_TOKEN: {Config.VNPT_ACCESS_TOKEN[:30]}... (len={len(Config.VNPT_ACCESS_TOKEN)})")
    print(f"   🤖 LLM_MODEL_SMALL: {Config.LLM_MODEL_SMALL}")
    print(f"   🤖 LLM_MODEL_LARGE: {Config.LLM_MODEL_LARGE}")
except Exception as e:
    print(f"   ❌ Config access error: {e}")
    traceback.print_exc()
    sys.exit(1)

# STEP 4: Check credentials
print("\n4️⃣ Checking credentials...")
try:
    creds_small = Config.VNPT_CREDENTIALS.get(Config.LLM_MODEL_SMALL)
    creds_large = Config.VNPT_CREDENTIALS.get(Config.LLM_MODEL_LARGE)
    
    if creds_small:
        print(f"   ✅ Small model creds found")
        print(f"      - token_id: {creds_small['token_id'][:20]}...")
        print(f"      - token_key: {creds_small['token_key'][:20]}...")
    else:
        print(f"   ❌ No credentials for {Config.LLM_MODEL_SMALL}")
    
    if creds_large:
        print(f"   ✅ Large model creds found")
        print(f"      - token_id: {creds_large['token_id'][:20]}...")
        print(f"      - token_key: {creds_large['token_key'][:20]}...")
    else:
        print(f"   ❌ No credentials for {Config.LLM_MODEL_LARGE}")
        
except Exception as e:
    print(f"   ❌ Credentials error: {e}")
    traceback.print_exc()
    sys.exit(1)

# STEP 5: Build URL and headers
print("\n5️⃣ Building request...")
try:
    model_name = Config.LLM_MODEL_SMALL
    creds = Config.VNPT_CREDENTIALS.get(model_name)
    
    url = f"{Config.VNPT_API_URL}/{model_name.replace('_', '-')}"
    print(f"   📍 URL: {url}")
    
    headers = {
        'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}',
        'Token-id': creds['token_id'],
        'Token-key': creds['token_key'],
        'Content-Type': 'application/json'
    }
    print(f"   ✅ Headers built")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Test"}],
        "temperature": 0.1,
        "max_completion_tokens": 50
    }
    print(f"   ✅ Payload built")
    
except Exception as e:
    print(f"   ❌ Build error: {e}")
    traceback.print_exc()
    sys.exit(1)

# STEP 6: Test actual API call
print("\n6️⃣ Testing API call...")

async def test_call():
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            print("   📤 Sending request...")
            
            async with session.post(url, json=payload, headers=headers) as resp:
                print(f"   📥 Status: {resp.status}")
                
                text = await resp.text()
                print(f"   📄 Response length: {len(text)} chars")
                
                if resp.status == 200:
                    try:
                        data = json.loads(text)
                        if 'choices' in data:
                            print(f"   ✅ SUCCESS!")
                            print(f"   💬 Answer: {data['choices'][0]['message']['content']}")
                        else:
                            print(f"   ⚠️ No 'choices' in response")
                            print(f"   Keys: {list(data.keys())}")
                    except json.JSONDecodeError as e:
                        print(f"   ❌ JSON decode error: {e}")
                        print(f"   Raw response: {text[:200]}")
                else:
                    print(f"   ❌ HTTP {resp.status}")
                    print(f"   Response: {text[:300]}")
                    
    except asyncio.TimeoutError:
        print(f"   ⏰ Timeout after 30s")
    except aiohttp.ClientError as e:
        print(f"   🔌 Connection error: {type(e).__name__}")
        print(f"   Details: {str(e)}")
    except Exception as e:
        print(f"   💥 Unexpected error: {type(e).__name__}")
        print(f"   Details: {str(e)}")
        traceback.print_exc()

try:
    asyncio.run(test_call())
except Exception as e:
    print(f"   ❌ asyncio.run failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Debug complete")
print("=" * 60)