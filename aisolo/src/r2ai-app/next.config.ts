import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Ẩn dev indicator (vòng tròn "N" góc màn hình) khi chạy dev.
  devIndicators: false,
  // Cho phép truy cập dev qua LAN/network IP (vd demo/test qua Chrome máy khác) — Next 16 chặn mặc định.
  allowedDevOrigins: ["192.168.1.50"],
  // @strands-agents/sdk có nhiều plugin phụ (context-offloader, bedrock, anthropic...) import động
  // các SDK optional (@aws-sdk/client-s3...) mà dự án không dùng và không cài. Webpack vẫn cố resolve
  // tĩnh các import động này khi build production → lỗi "module not found" dù code path không bao
  // giờ chạy tới. Đánh dấu external để webpack bỏ qua lúc bundle, Node tự resolve lúc chạy (không
  // bao giờ bị gọi tới vì chỉ dùng OpenAIModel + GoalLoop, không dùng context-offloader/bedrock).
  serverExternalPackages: ["@strands-agents/sdk"],
};

export default nextConfig;
