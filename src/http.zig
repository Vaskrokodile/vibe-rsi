const std = @import("std");
const util = @import("util.zig");

pub const Header = std.http.Header;

pub const Response = struct {
    status: std.http.Status,
    body: []u8,
};

pub const HttpClient = struct {
    client: std.http.Client,

    pub fn init(alloc: std.mem.Allocator) HttpClient {
        var c: std.http.Client = .{ .allocator = alloc };
        c.initDefaultProxies(alloc) catch {};
        return .{ .client = c };
    }

    pub fn deinit(self: *HttpClient) void {
        self.client.deinit();
    }

    /// One-shot request with buffered response body.
    pub fn fetch(
        self: *HttpClient,
        alloc: std.mem.Allocator,
        method: std.http.Method,
        url: []const u8,
        extra_headers: []const Header,
        payload: ?[]const u8,
    ) !Response {
        var body = std.ArrayList(u8).init(alloc);
        errdefer body.deinit();
        var header_buf: [16 * 1024]u8 = undefined;
        var req = try self.open(method, try std.Uri.parse(url), .{
            .server_header_buffer = &header_buf,
            .redirect_behavior = if (payload != null) .unhandled else @enumFromInt(3),
            .headers = .{ .content_type = .{ .override = "application/json" } },
            .extra_headers = extra_headers,
        });
        defer req.deinit();
        if (payload) |data| req.transfer_encoding = .{ .content_length = data.len };
        try req.send();
        if (payload) |data| try req.writeAll(data);
        try req.finish();
        try req.wait();
        try req.reader().readAllArrayList(&body, 128 << 20);
        return .{ .status = req.response.status, .body = try body.toOwnedSlice() };
    }

    /// Plain GET for arbitrary web content — no forced content-type.
    pub fn fetchUrl(
        self: *HttpClient,
        alloc: std.mem.Allocator,
        url: []const u8,
        extra_headers: []const Header,
        max_bytes: usize,
    ) !Response {
        var body = std.ArrayList(u8).init(alloc);
        errdefer body.deinit();
        var header_buf: [16 * 1024]u8 = undefined;
        var req = try self.open(.GET, try std.Uri.parse(url), .{
            .server_header_buffer = &header_buf,
            .extra_headers = extra_headers,
        });
        defer req.deinit();
        try req.send();
        try req.finish();
        try req.wait();
        try req.reader().readAllArrayList(&body, max_bytes);
        return .{ .status = req.response.status, .body = try body.toOwnedSlice() };
    }

    pub fn open(
        self: *HttpClient,
        method: std.http.Method,
        uri: std.Uri,
        options: std.http.Client.RequestOptions,
    ) !std.http.Client.Request {
        const scheme = uri.scheme;
        const protocol: std.http.Client.Connection.Protocol = if (std.ascii.eqlIgnoreCase(scheme, "https"))
            .tls
        else if (std.ascii.eqlIgnoreCase(scheme, "http"))
            .plain
        else
            return error.UnsupportedScheme;
        const proxy = if (protocol == .tls) self.client.https_proxy else self.client.http_proxy;
        if (proxy != null) return self.client.open(method, uri, options);

        const host = try (uri.host orelse return error.UriMissingHost).toRawMaybeAlloc(self.client.allocator);
        const port = uri.port orelse if (protocol == .tls) @as(u16, 443) else 80;
        const criteria = std.http.Client.ConnectionPool.Criteria{ .host = host, .port = port, .protocol = protocol };
        if (self.client.connection_pool.findConnection(criteria)) |connection| {
            var options_with_connection = options;
            options_with_connection.connection = connection;
            return self.client.open(method, uri, options_with_connection) catch |err| {
                self.client.connection_pool.release(self.client.allocator, connection);
                return err;
            };
        }

        if (protocol == .tls) try self.ensureCertificates();
        const connection = try self.connectHostV4First(host, port, protocol);
        var options_with_connection = options;
        options_with_connection.connection = connection;
        return self.client.open(method, uri, options_with_connection) catch |err| {
            self.client.connection_pool.release(self.client.allocator, connection);
            return err;
        };
    }

    fn ensureCertificates(self: *HttpClient) !void {
        if (std.http.Client.disable_tls or !@atomicLoad(bool, &self.client.next_https_rescan_certs, .acquire)) return;
        self.client.ca_bundle_mutex.lock();
        defer self.client.ca_bundle_mutex.unlock();
        if (!self.client.next_https_rescan_certs) return;
        try self.client.ca_bundle.rescan(self.client.allocator);
        @atomicStore(bool, &self.client.next_https_rescan_certs, false, .release);
    }

    fn connectHostV4First(
        self: *HttpClient,
        host: []const u8,
        port: u16,
        protocol: std.http.Client.Connection.Protocol,
    ) !*std.http.Client.Connection {
        const alloc = self.client.allocator;
        var addresses = try std.net.getAddressList(alloc, host, port);
        defer addresses.deinit();
        if (addresses.addrs.len == 0) return error.UnknownHostName;

        var last_error: anyerror = error.ConnectionRefused;
        for (0..2) |pass| {
            for (addresses.addrs) |address| {
                const is_v4 = address.any.family == std.posix.AF.INET;
                if ((pass == 0) != is_v4) continue;
                const stream = std.net.tcpConnectToAddress(address) catch |err| {
                    last_error = err;
                    continue;
                };
                return try self.registerConnection(stream, host, port, protocol);
            }
        }
        return last_error;
    }

    fn registerConnection(
        self: *HttpClient,
        stream: std.net.Stream,
        host: []const u8,
        port: u16,
        protocol: std.http.Client.Connection.Protocol,
    ) !*std.http.Client.Connection {
        const alloc = self.client.allocator;
        const node = try alloc.create(std.http.Client.ConnectionPool.Node);
        errdefer alloc.destroy(node);
        node.* = .{ .data = undefined };
        errdefer stream.close();
        node.data = .{
            .stream = stream,
            .tls_client = undefined,
            .protocol = protocol,
            .host = try alloc.dupe(u8, host),
            .port = port,
        };
        errdefer alloc.free(node.data.host);

        if (protocol == .tls) {
            if (std.http.Client.disable_tls) return error.TlsInitializationFailed;
            node.data.tls_client = try alloc.create(std.crypto.tls.Client);
            errdefer alloc.destroy(node.data.tls_client);
            node.data.tls_client.* = try std.crypto.tls.Client.init(stream, .{
                .host = .{ .explicit = host },
                .ca = .{ .bundle = self.client.ca_bundle },
                .ssl_key_log_file = null,
            });
            node.data.tls_client.allow_truncation_attacks = true;
        }

        self.client.connection_pool.addUsed(node);
        return &node.data;
    }
};

/// Streaming POST that yields each SSE `data:` payload to a callback.
/// Returns the response status. The callback receives raw `data:` payloads
/// (without the "data: " prefix); "[DONE]" is passed through like any other.
pub const SseContext = struct {
    ctx: *anyopaque,
    on_data: *const fn (ctx: *anyopaque, data: []const u8) anyerror!void,
};

pub fn postSse(
    http: *HttpClient,
    alloc: std.mem.Allocator,
    url: []const u8,
    extra_headers: []const Header,
    payload: []const u8,
    sse: SseContext,
) !std.http.Status {
    const uri = try std.Uri.parse(url);
    var hdr_buf: [32 * 1024]u8 = undefined;
    var req = try http.open(.POST, uri, .{
        .server_header_buffer = &hdr_buf,
        .headers = .{ .content_type = .{ .override = "application/json" } },
        .extra_headers = extra_headers,
    });
    defer req.deinit();
    req.transfer_encoding = .{ .content_length = payload.len };
    try req.send();
    try req.writer().writeAll(payload);
    try req.finish();
    try req.wait();

    if (req.response.status != .ok) {
        // drain a bit of the error body for diagnostics
        var err_buf: [4096]u8 = undefined;
        const n = req.reader().read(&err_buf) catch 0;
        if (n > 0) {
            std.debug.print("{s}\n", .{err_buf[0..n]});
        }
        return req.response.status;
    }

    const reader = req.reader();
    var pending = std.ArrayList(u8).init(alloc);
    defer pending.deinit();

    while (try reader.readUntilDelimiterOrEofAlloc(alloc, '\n', 4 << 20)) |line_raw| {
        var line = line_raw;
        if (line.len > 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
        if (line.len == 0) {
            if (pending.items.len > 0) {
                try sse.on_data(sse.ctx, pending.items);
                pending.clearRetainingCapacity();
            }
            continue;
        }
        if (std.mem.startsWith(u8, line, "data:")) {
            var d = line[5..];
            if (d.len > 0 and d[0] == ' ') d = d[1..];
            if (pending.items.len > 0) try pending.append('\n');
            try pending.appendSlice(d);
        }
        // ignore event:/id:/retry:/comment lines
    }
    if (pending.items.len > 0) try sse.on_data(sse.ctx, pending.items);
    return req.response.status;
}
