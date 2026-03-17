import requests
import json
import time
from typing import List, Dict, Any


def measure_streaming_performance(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:

    payload_json = json.dumps(payload)
    start_time = time.time()
    response = requests.post(url, headers=headers, data=payload_json, stream=True)
    if response.status_code != 200:
        return {
            "error": f"{response.status_code}",
            "response_text": response.text
        }

    first_token_time = None
    token_times = []
    tokens = []

    try:
        for line in response.iter_lines(decode_unicode=True):
            current_time = time.time()

            if line:
                if first_token_time is None:
                    first_token_time = current_time
                token_times.append(current_time)
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'usage' in data:
                            tokens.append(data['usage']['completion_tokens'])
                        if 'data' in data:
                            tokens.append(data['data'][0]['data'])
                    except json.JSONDecodeError:
                        tokens.append(line[6:] if line.startswith('data: ') else line)
                else:
                    tokens.append(line)

    except Exception as e:
        return {
            "error": f"Error{str(e)}",
            "partial_tokens": tokens,
            "partial_times": token_times
        }

    end_time = time.time()
    metrics = calculate_metrics(start_time, first_token_time, token_times, end_time, tokens)
    return metrics


def calculate_metrics(start_time: float, first_token_time: float, token_times: List[float],
                      end_time: float, tokens: List[str]) -> Dict[str, Any]:

    if first_token_time is None:
        return {
            "error": "No tokens received",
            "overall_time": end_time - start_time
        }
    ttft = first_token_time - start_time
    overall_time = end_time - start_time
    inter_token_latencies = []
    if len(token_times) > 1:
        for i in range(1, len(token_times)):
            inter_token_latencies.append(token_times[i] - token_times[i - 1])
    total_tokens = len(tokens)

    metrics = {
        "ttft_seconds": round(ttft, 4),
        "ttft_ms": round(ttft * 1000, 2),
        "overall_time_seconds": round(overall_time, 4),
        "overall_time_ms": round(overall_time * 1000, 2),
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_tokens / overall_time, 2) if overall_time > 0 else 0,
        "inter_token_latencies_ms": [round(lat * 1000, 2) for lat in inter_token_latencies],
        "avg_inter_token_latency_ms": round(sum(inter_token_latencies) * 1000 / len(inter_token_latencies),
                                            2) if inter_token_latencies else 0,
        "min_inter_token_latency_ms": round(min(inter_token_latencies) * 1000, 2) if inter_token_latencies else 0,
        "max_inter_token_latency_ms": round(max(inter_token_latencies) * 1000, 2) if inter_token_latencies else 0,
        "tokens": tokens[:5] if len(tokens) > 5 else tokens
    }

    return metrics


def print_metrics(metrics: Dict[str, Any]):
    if "error" in metrics:
        print(f"❌ Error: {metrics['error']}")
        return

    print("📊 Streaming Performance Metrics")
    print("=" * 40)
    print(f"🚀 Time to First Token (TTFT): {metrics['ttft_ms']} ms ({metrics['ttft_seconds']} seconds)")
    print(f"⏱️  Overall Time: {metrics['overall_time_ms']} ms ({metrics['overall_time_seconds']} seconds)")
    print(f"🔢 Total Tokens: {metrics['total_tokens']}")
    print(f"📈 Tokens per Second: {metrics['tokens_per_second']}")
    print()
    print("🔄 Inter-Token Latency Statistics:")
    print(f"   Average: {metrics['avg_inter_token_latency_ms']} ms")
    print(f"   Minimum: {metrics['min_inter_token_latency_ms']} ms")
    print(f"   Maximum: {metrics['max_inter_token_latency_ms']} ms")
    print()
    print(f"📝 Sample Tokens: {metrics['tokens']}")


url = "https://crmintelligencepy-lab.kites.localzoho.com/llm/text/api/qwentext/quantized/generate"

payload = {
    "prompt": "hiii, tell me a short info about COC Game. ",
    "max_tokens": 12000,
    "temperature": 0,
    "stream": True
}

headers = {
    'Content-Type': 'application/json',
    'Cookie': 'zalb_791c15dd5f=f0b83a68b20b8687864dc215f54d0554'
}

if __name__ == "__main__":
    print("🔄 Starting streaming performance measurement...")
    print(f"📡 URL: {url}")
    print(f"📝 Prompt: {payload['prompt']}")
    print()

    # Measure performance
    metrics = measure_streaming_performance(url, headers, payload)

    # Print results
    print_metrics(metrics)

    # Optional: Save metrics to file
    with open('streaming_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\n💾 Metrics saved to 'streaming_metrics.json'")
