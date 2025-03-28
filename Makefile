clean:
	rm -rf ./external/opencv-4.11.0/build/
	rm -rf ./build/
	rm -rf ./external/opencv/


fmt:
	clang-format -i include/*.h -i src/* -style='Mozilla'

