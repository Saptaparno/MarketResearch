.PHONY: eval eval-accuracy eval-reliability eval-performance eval-judge

eval:
	cd tests/evals && python run_all.py

eval-accuracy:
	cd tests/evals && python run_all.py accuracy

eval-reliability:
	cd tests/evals && python run_all.py reliability

eval-performance:
	cd tests/evals && python run_all.py performance

eval-judge:
	cd tests/evals && python run_all.py agent_as_judge
