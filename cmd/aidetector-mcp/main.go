// Command aidetector-mcp serves the AI text detector over MCP (stdio).
package main

import (
	"context"
	"log"
	"os"
	"os/signal"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/joeshirey/AIDetector/internal/engine"
	"github.com/joeshirey/AIDetector/internal/mcpserver"
)

var version = "dev"

func main() {
	log.SetOutput(os.Stderr) // stdout carries the MCP protocol
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	e := engine.Default()
	v := version
	if v == "dev" {
		if bv := e.BuildInfo().Version; bv != "" {
			v = bv
		}
	}
	if err := mcpserver.New(e, v).Run(ctx, &mcp.StdioTransport{}); err != nil {
		log.Fatal(err)
	}
}
