import { handleUpload, type HandleUploadBody } from "@vercel/blob/client";
import { NextResponse } from "next/server";

export async function POST(request: Request): Promise<NextResponse> {
  const traceId = crypto.randomUUID();
  let body: HandleUploadBody;
  try {
    body = (await request.json()) as HandleUploadBody;
  } catch (error) {
    console.error("[upload-token] invalid json body", {
      traceId,
      error: error instanceof Error ? error.message : String(error),
    });
    return NextResponse.json({ error: "Invalid upload payload", traceId }, { status: 400 });
  }

  const hasBlobToken = Boolean(process.env.BLOB_READ_WRITE_TOKEN);
  const hasBlobStoreId = Boolean(process.env.BLOB_STORE_ID);
  const bodyInfo = {
    pathname: (body as { pathname?: string }).pathname,
    contentType: (body as { contentType?: string }).contentType,
  };

  try {
    const jsonResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (params) => {
        console.info("[upload-token] token-params", {
          traceId,
          pathname: params.pathname,
          contentType: params.contentType,
          clientPayload: params.clientPayload,
        });
        // Some browsers/filesystems send PDFs as application/octet-stream.
        // Allow both so valid PDF uploads don't fail with Blob 400.
        return {
          allowedContentTypes: ["application/pdf", "application/octet-stream"],
          maximumSizeInBytes: 50 * 1024 * 1024,
        };
      },
      onUploadCompleted: async () => {},
    });
    console.info("[upload-token] success", {
      traceId,
      pathname: new URL(request.url).pathname,
      hasBlobToken,
      hasBlobStoreId,
      bodyInfo,
    });
    return NextResponse.json(jsonResponse, {
      headers: { "x-trace-id": traceId },
    });
  } catch (error) {
    console.error("[upload-token] failed", {
      traceId,
      pathname: new URL(request.url).pathname,
      hasBlobToken,
      hasBlobStoreId,
      bodyInfo,
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Upload token generation failed", traceId },
      { status: 400, headers: { "x-trace-id": traceId } },
    );
  }
}
