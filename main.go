package main

import (
	"embed"
	"io/fs"
	"log"
	"os"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	app := NewApp(findRoot())
	frontend, err := fs.Sub(assets, "frontend/dist")
	if err != nil {
		log.Fatal(err)
	}
	if len(os.Args) > 1 && os.Args[1] == "--serve" {
		serve(app, frontend)
		return
	}
	err = wails.Run(&options.App{
		Title: "OpWeaver Studio · 算织", Width: 1560, Height: 960, MinWidth: 960, MinHeight: 640,
		BackgroundColour: &options.RGBA{R: 14, G: 17, B: 23, A: 255},
		AssetServer:      &assetserver.Options{Assets: frontend},
		OnStartup:        app.startup, OnShutdown: app.shutdown, Bind: []interface{}{app},
	})
	if err != nil {
		log.Fatal(err)
	}
}
