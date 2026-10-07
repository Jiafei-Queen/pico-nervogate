# syntax=docker/dockerfile:1
FROM rust:1.92-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends musl-tools \
    && rm -rf /var/lib/apt/lists/*
RUN rustup target add aarch64-unknown-linux-musl
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --target aarch64-unknown-linux-musl

FROM scratch
COPY --from=build /app/target/aarch64-unknown-linux-musl/release/pico-nervogate /pico-nervogate
COPY examples/gateway.min.toml /gateway.toml
EXPOSE 8787
ENTRYPOINT ["/pico-nervogate"]
