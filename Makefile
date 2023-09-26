

setup:
	@echo "Settin up"
	@echo "Installing PIPENV package..."
	python -m pip install pipenv
	@echo "Installing dependencies..."
	pipenv install
	@echo "####DONE####"
task1:
	@echo "####TASK 1####"
	@echo "Running task 1"
	pipenv run python ./homeworkSolutions/task1/linePoint.py
	@echo "Initialising test process"
	pipenv run pytest ./test/testPointGP2.py -v -rs
	@echo "####DONE####"
task2:
	@echo "####TASK 2####"
	@echo "Running task 2"
	pipenv run python ./homeworkSolutions/task2/conical.py
	@echo "####DONE####"
task3:
	@echo "####TASK 3####"
	@echo "Running task 3"
	pipenv run python ./homeworkSolutions/task3/conical.py
	@echo "####DONE####"
task4:
	@echo "####TASK 4####"
	@echo "Running task 4"
	pipenv run python ./homeworkSolutions/task4/interpolationExamples.py
	@echo "####DONE####"
task5:
	@echo "####TASK 5####"
	@echo "Running task 5"
	@echo "Image task"
	pipenv run python ./homeworkSolutions/task5/openImage.py
	@echo "Finished image task"
	@echo "Video task"
	pipenv run python ./homeworkSolutions/task5/openVideo.py
	@echo "Finished video task"
	@echo "####DONE####"
task6:
	@echo "####TASK 6####"
	@echo "Running task 6"
	pipenv run python ./homeworkSolutions/task6/homographyTask.py
	@echo "####DONE####"