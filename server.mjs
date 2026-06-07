import { createReadStream, existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { extname, join, normalize } from "node:path";

const root = process.cwd();
const port = Number(process.env.PORT ?? 5173);
const globalDataPath = join(root, "data", "global-data.json");
loadEnvFile();
const adminUser = process.env.ADMIN_USER ?? "";
const adminPassword = process.env.ADMIN_PASSWORD ?? "";
const adminToken = process.env.ADMIN_TOKEN ?? randomUUID();
const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8"
};

mkdirSync(join(root, "data"), { recursive: true });

createServer(async (request, response) => {
  const url = new URL(request.url ?? "/", `http://${request.headers.host}`);
  if (url.pathname.startsWith("/api/")) {
    await handleApi(request, response, url);
    return;
  }

  const requestedPath = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, "");
  let filePath = join(root, requestedPath);

  if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
    filePath = join(root, "index.html");
  }

  response.setHeader("Content-Type", types[extname(filePath)] ?? "application/octet-stream");
  createReadStream(filePath).pipe(response);
}).listen(port, "127.0.0.1", () => {
  console.log(`VInculacion listo en http://localhost:${port}`);
});

async function handleApi(request, response, url) {
  try {
    if (request.method === "GET" && url.pathname === "/api/global-data") {
      sendJson(response, 200, await readGlobalData());
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/admin-login") {
      const body = await readJsonBody(request);
      if (!adminUser || !adminPassword) {
        sendJson(response, 503, { error: "Credenciales de administrador no configuradas" });
        return;
      }
      if (body.username === adminUser && body.password === adminPassword) {
        sendJson(response, 200, { token: adminToken });
      } else {
        sendJson(response, 401, { error: "Credenciales incorrectas" });
      }
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/global-data") {
      if (!isAdmin(request)) {
        sendJson(response, 401, { error: "Solo administradores pueden guardar globalmente" });
        return;
      }

      const body = await readJsonBody(request);
      const key = body.type === "letters" ? "letters" : "signs";
      const item = body.item;
      if (!item?.label || !Array.isArray(item.samples)) {
        sendJson(response, 400, { error: "Dato global invalido" });
        return;
      }

      const data = await readGlobalData();
      const existing = data[key].find((entry) => entry.label === item.label);
      if (existing) {
        existing.samples.push(...item.samples);
      } else {
        data[key].push(item);
      }
      await writeGlobalData(data);
      sendJson(response, 200, data);
      return;
    }

    if (request.method === "DELETE" && url.pathname === "/api/global-data") {
      if (!isAdmin(request)) {
        sendJson(response, 401, { error: "Solo administradores pueden borrar globalmente" });
        return;
      }

      const key = url.searchParams.get("type") === "letters" ? "letters" : "signs";
      const label = url.searchParams.get("label");
      const data = await readGlobalData();
      data[key] = data[key].filter((entry) => entry.label !== label);
      await writeGlobalData(data);
      sendJson(response, 200, data);
      return;
    }

    sendJson(response, 404, { error: "Ruta no encontrada" });
  } catch (error) {
    console.error(error);
    sendJson(response, 500, { error: "Error del servidor" });
  }
}

async function readGlobalData() {
  try {
    const raw = await readFile(globalDataPath, "utf8");
    const data = JSON.parse(raw);
    return {
      signs: Array.isArray(data.signs) ? data.signs : [],
      letters: Array.isArray(data.letters) ? data.letters : []
    };
  } catch {
    return { signs: [], letters: [] };
  }
}

async function writeGlobalData(data) {
  await writeFile(globalDataPath, `${JSON.stringify(data, null, 2)}\n`, "utf8");
}

function isAdmin(request) {
  return request.headers.authorization === `Bearer ${adminToken}`;
}

function readJsonBody(request) {
  return new Promise((resolve, reject) => {
    let raw = "";
    request.on("data", (chunk) => {
      raw += chunk;
      if (raw.length > 25_000_000) request.destroy();
    });
    request.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch (error) {
        reject(error);
      }
    });
    request.on("error", reject);
  });
}

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

function loadEnvFile() {
  const envPath = join(root, ".env");
  if (!existsSync(envPath)) return;

  const lines = readFileSync(envPath, "utf8").split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
    const [key, ...valueParts] = trimmed.split("=");
    if (!process.env[key]) {
      process.env[key] = valueParts.join("=").trim();
    }
  }
}
