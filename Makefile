.PHONY: run api test bench docker
run:    ; ./run.sh
api:    ; ./run.sh api
test:   ; ./run.sh test
bench:  ; ./run.sh bench
docker: ; docker compose up --build
