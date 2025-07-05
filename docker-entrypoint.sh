#!/bin/bash

# Wait for RabbitMQ to be ready
echo "Waiting for RabbitMQ to be ready..."
while ! curl -f http://rabbitmq:15672/api/overview >/dev/null 2>&1; do
    echo "Waiting for RabbitMQ..."
    sleep 2
done
echo "RabbitMQ is ready!"

# If no arguments provided, start interactive shell
if [ $# -eq 0 ]; then
    echo "=========================================="
    echo "🚀 StockSim Docker Container Ready!"
    echo "=========================================="
    echo ""
    echo "Available commands:"
    echo "  📊 Run simulation:     python main_launcher.py configs/demo_config.yaml"
    echo "  📈 Generate charts:    python utils/plot_charts.py --help"
    echo "  ⚙️  Edit configs:       nano configs/demo_config.yaml"
    echo "  📁 List configs:       ls configs/"
    echo "  📋 View logs:          tail -f logs/*.log"
    echo "  🔍 Monitor resources:  htop"
    echo ""
    echo "Volumes mounted:"
    echo "  📁 configs/  -> Editable configuration files"
    echo "  📁 charts/   -> Generated charts output"
    echo "  📁 reports/  -> Simulation reports"
    echo "  📁 logs/     -> Application logs"
    echo ""
    exec bash
fi

# Execute the provided command
exec "$@"