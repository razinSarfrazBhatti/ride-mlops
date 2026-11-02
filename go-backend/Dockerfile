# Build stage
FROM golang:1.24-alpine AS builder

WORKDIR /app

COPY go.mod ./
COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -o server .

# Runtime stage
FROM alpine:3.21

WORKDIR /app

COPY --from=builder /app/server .

EXPOSE 8081

CMD ["./server"]
