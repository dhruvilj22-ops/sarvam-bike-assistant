import { put } from "@vercel/blob";
import { NextResponse } from "next/server";

export async function POST(request: Request): Promise<NextResponse> {
  const traceId = crypto.randomUUID();

  try {
    const formData = await request.formData();
    const file = formData.get("file");

    if (!(file instanceof File)) {
      return NextResponse.json(
        { error: "Missing file in multipart payload", traceId },
        { status: 400, headers: { "x-trace-id": traceId } },
      );
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      return NextResponse.json(
        { error: "Only PDF files are supported", traceId },
        { status: 400, headers: { "x-trace-id": traceId } },
      );
    }

    const blob = await put(file.name, file, {
      access: "public",
      addRandomSuffix: false,
      contentType: file.type || "application/pdf",
    });

    console.info("[upload-file] success", {
      traceId,
      filename: file.name,
      size: file.size,
      contentType: file.type,
      blobUrl: blob.url,
    });

    return NextResponse.json(
      { url: blob.url },
      { status: 200, headers: { "x-trace-id": traceId } },
    );
  } catch (error) {
    console.error("[upload-file] failed", {
      traceId,
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });

    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Upload failed", traceId },
      { status: 500, headers: { "x-trace-id": traceId } },
    );
  }
}
