.PHONY: test dry-run report
test:        ## run everything that does not need a GPU (E1-E12 + units)
	python -m pytest tests -q

dry-run:     ## full A/B harness with mock LLM + mock runner
	python -m src.mock_repo evals/fixtures/mock_git_repo
	python -m evals.run_ab --config evals/fixtures/dry_config.yaml --repeats 1 --out /tmp/ab_out

report:      ## regenerate results/ab_report.md + plots
	python -m evals.report results
