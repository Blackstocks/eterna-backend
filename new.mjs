import { PythLazerClient } from "@pythnetwork/pyth-lazer-sdk";
import { createClient } from "redis";

// Simple logger 
const log = {
  info: (m) => console.log(`[INFO] ${new Date().toISOString()} - ${m}`),
  warn: (m) => console.warn(`[WARN] ${new Date().toISOString()} - ${m}`),
  error: (m) => console.error(`[ERROR] ${new Date().toISOString()} - ${m}`),
  debug: (m, ...rest) => console.debug(`[DEBUG] ${new Date().toISOString()} - ${m}`, ...rest),
};

const LAZER_URL = "wss://pyth-lazer.dourolabs.app/v1/stream"; // prod endpoint [2]
const LAZER_TOKEN = process.env.PYTH_LAZER_TOKEN || "gr6DS1uhFL7dUrcrboueU4ykRk2XhOfT3GO-demo"; // demo token
const PRICE_FEED_IDS = [1]; // BTC/USD per your snippet; ensure it matches your account’s allowed IDs [2]
const REDIS_URL = process.env.REDIS_URL || "redis://127.0.0.1:6379";
const REDIS_CHANNEL = "prices.btcusd";
 
const main = async () => {
  try {
    log.info("Connecting to Redis...");
    const redis = createClient({ url: REDIS_URL });
    redis.on("error", (err) => log.error(`Redis error: ${err.message}`));
    await redis.connect();
    log.info("Connected to Redis");

    log.info("Starting Pyth Lazer client connection...");
    const client = await PythLazerClient.create({
      urls: [LAZER_URL],
      token: LAZER_TOKEN, // auth required for /v1/stream [2]
    });
    log.info("Successfully connected to Pyth Lazer");

    // Subscribe using explicit deliveryFormat and parsed to simplify JSON handling [2]
    client.subscribe({
      type: "subscribe",
      subscriptionId: 1,
      priceFeedIds: PRICE_FEED_IDS,
      properties: ["price", "bestBidPrice", "bestAskPrice", "exponent"],
      chains: ["solana"],
      deliveryFormat: "json",
      channel: "real_time",
      parsed: true,
      jsonBinaryEncoding: "hex",
    });

    log.info("Subscribed to BTC/USD real-time feed");

    client.addMessageListener(async (message) => {
      try {
        if (message.type === "json") {
          const v = message.value;
          if (v?.type === "streamUpdated") {
            const pf = v.parsed?.priceFeeds;
            if (Array.isArray(pf)) {
              for (const feed of pf) {
                if (feed?.priceFeedId === 1) { // matches subscription [2]
                  // Lazer sends integers with exponent; convert to float
                  const exp = typeof feed.exponent === "number" ? feed.exponent : -8; // fallback like your example
                  const scale = Math.pow(10, -exp);
                  const priceNum = feed.price != null ? Number(feed.price) / scale : null;
                  const bidNum = feed.bestBidPrice != null ? Number(feed.bestBidPrice) / scale : null;
                  const askNum = feed.bestAskPrice != null ? Number(feed.bestAskPrice) / scale : null;
                  const tsMicros = v.timestamp; // microsecond precision per docs [2]
                  const ts = tsMicros != null ? Math.floor(Number(tsMicros) / 1_000_000) : Math.floor(Date.now() / 1000);

                  const tick = { ts, price: priceNum, bid: bidNum, ask: askNum };
                  await redis.publish(REDIS_CHANNEL, JSON.stringify(tick));
                  log.debug(`Published tick to ${REDIS_CHANNEL}:`, tick);
                }
              }
            } else {
              log.warn("No parsed.priceFeeds in streamUpdated");
            }
          } else if (v?.type === "subscribed") {
            log.info(`Subscribed ack for subscriptionId=${v.subscriptionId}`);
          } else if (v?.type === "subscriptionError") {
            log.error(`Subscription error: ${v.message || "unknown"}`);
          } else {
            log.debug(`Received json message type: ${v?.type}`);
          }
        } else if (message.type === "binary") {
          // You can parse binary if needed; skipped here
        } else {
          log.warn(`Unknown message type: ${message.type}`);
        }
      } catch (e) {
        log.error(`Listener error: ${e.message}`);
      }
    });

    client.addAllConnectionsDownListener(() => {
      log.warn("All Lazer connections are down - attempting reconnect...");
    });

    log.info("✅ Connected to Pyth Lazer - Streaming BTC prices in real-time...");
    log.info("⏱️ Forwarding price updates to Redis Pub/Sub...");

    // Optional: shutdown after N minutes
    const RUN_MS = Number(process.env.RUN_MS || 0);
    if (RUN_MS > 0) {
      setTimeout(async () => {
        log.info("Shutting down...");
        client.shutdown();
        await redis.quit();
        process.exit(0);
      }, RUN_MS);
    }
  } catch (error) {
    log.error(`Connection failed: ${error.message}`);
    log.error(`Stack trace: ${error.stack}`);
    process.exit(1);
  }
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
