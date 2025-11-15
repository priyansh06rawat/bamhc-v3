export async function classifyText(text) {
  const body = JSON.stringify({ text });
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body, signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || "API error");
    }
    return await res.json();
  } finally {
    clearTimeout(timeout);
  }
}
