.DEFAULT_GOAL := help

.PHONY: help status pull add commit push sync dsstore-clean

BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
REMOTE ?= origin
MSG ?=
AUTO_MSG ?= chore: auto sync ($(shell date "+%Y-%m-%d %H:%M:%S"))

help:
	@echo "ITPHYS Git helper (Makefile)"
	@echo ""
	@echo "Usage:"
	@echo "  make status"
	@echo "  make pull"
	@echo "  make add"
	@echo "  make commit MSG=\"your message\""
	@echo "  make push"
	@echo "  make sync   (pull --rebase then push)"
	@echo "  make dsstore-clean"
	@echo ""
	@echo "Vars:"
	@echo "  REMOTE=origin (default: origin)"
	@echo "  MSG=...       (required for commit)"
	@echo ""
	@echo "Current branch: $(BRANCH)"

status:
	@git status

pull:
	@git pull $(REMOTE) $(BRANCH)

add:
	@git add -A

commit:
	@if [ -z "$(MSG)" ]; then \
		echo "ERROR: commit message is required. Example: make commit MSG=\"update report\""; \
		exit 1; \
	fi
	@git commit -m "$(MSG)"

push:
	@$(MAKE) dsstore-clean
	@$(MAKE) add
	@# If there is anything staged, require MSG and commit it.
	@if git diff --cached --quiet; then \
		echo "No staged changes to commit. Pushing current branch..."; \
	else \
		if [ -z "$(MSG)" ]; then \
			echo "No MSG provided. Using AUTO_MSG: $(AUTO_MSG)"; \
			git commit -m "$(AUTO_MSG)"; \
		else \
			git commit -m "$(MSG)"; \
		fi; \
	fi
	@git push -u $(REMOTE) $(BRANCH)

sync:
	@git pull --rebase $(REMOTE) $(BRANCH)
	@git push $(REMOTE) $(BRANCH)

dsstore-clean:
	@echo "Removing .DS_Store files (working tree only)."
	@find . -name ".DS_Store" -print -delete
