# StockSim Docker Commands

# Build and start everything
build:
	docker-compose build

up:
	docker-compose up -d

# Interactive access to simulation container
shell:
	docker exec -it stocksim-simulation bash

# Run simulation with specified config (default: demo_config.yaml)
run:
	docker exec -it stocksim-simulation python main_launcher.py configs/$(or $(CONFIG),demo_config.yaml)

# Monitoring commands
monitor:
	docker stats stocksim-simulation stocksim-rabbitmq

logs:
	docker-compose logs -f stocksim

rabbitmq-logs:
	docker-compose logs -f rabbitmq

# Resource monitoring during simulation
monitor-csv:
	@echo "Logging container stats to monitor.log..."
	docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" --no-stream >> monitor.log

# Utility commands
charts:
	docker exec -it stocksim-simulation python utils/plot_charts.py --help

edit-config:
	docker exec -it stocksim-simulation nano configs/demo.yaml

# Cleanup
down:
	docker-compose down

clean:
	docker-compose down -v
	docker system prune -f

# Show current resource usage
stats:
	docker stats --no-stream

.PHONY: build up shell run monitor logs rabbitmq-logs monitor-csv charts edit-config down clean stats