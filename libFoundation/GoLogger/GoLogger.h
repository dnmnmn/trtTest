#ifndef __GO_LOGGER_H__
#define __GO_LOGGER_H__


#include <thread>
#include <atomic>
#include <mutex>
#include <list>
#include "../Library/onetbb_win/include/tbb/concurrent_queue.h"
#include <string>
#include <vector>
#include <iostream>
#include "../DateTime/DateTime.h"
#include "../FileSystem/FileSystem.h"

using namespace tbb::detail::d2;

enum LogLevel
{
	DEBUG = 0,
	INFO,
	WARN,
	ERROR,
	FATAL
};

class GoLogger
{
private:
	concurrent_queue<std::string>* log_queue = nullptr;
	std::thread m_thread;
	std::atomic<bool> m_running = false;

	DateTime timer;
	std::vector<string> log_levels = {"[DEBUG] ", "[INFO]  ", "[WARN]  ", "[ERROR] ", "[FATAL] "};
	int verbosity = 2;

	GoLogger() { log_queue = nullptr; }
	GoLogger(const GoLogger& other);
	// Logger& operator=(const Logger& other) {};
	~GoLogger() {};
	
public:
	static GoLogger& GetInstance()
	{
		static GoLogger instance;
		return instance;
	}
	void initialize(string path = "//log", int log_level = 1)
	{
		bool expect = false;
		if (m_running.compare_exchange_strong(expect, true))
		{
			verbosity = log_level;
			path = FileSystem::get_app_path() + path;
			if(!FileSystem::exist(path)) FileSystem::make_folder(path);
			log_queue = new concurrent_queue<std::string>();
			timer.now();
			path = path + "//" + timer.get_date_string();
			std::cout << path << std::endl;
			// thread start
			{
				std::string message = "\n==================================";
				message.append("Log Level : ");
				message.append(log_levels[verbosity]);
				message.append("==================================\n");
				timer.now();
				message.append(timer.get_log_string());
				message.append("\GoLogger started\n");
				log_queue->push(message);
			}
			m_thread = std::thread(&GoLogger::log_kernnel, this, path);
		}
	}
	void release()
	{
		m_running.exchange(false);
		m_thread.join();
		delete log_queue;
	}
	void log_kernnel(string path)
	{
		// thread loop
		while(m_running)
		{
			std::string message;
			if(log_queue->try_pop(message))
			{
				std::cout << message << std::endl;
				FileSystem::write_all_text(path, message);
			}
			else std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		// thread end
		while(!log_queue->empty())
		{
			std::string message;
			log_queue->try_pop(message);
			std::cout << message << std::endl;
			FileSystem::write_all_text(path, message);
		}
		{
			std::string message = "";
			timer.now();
			message.append(timer.get_log_string());
			message.append("\GoLogger stopped\n");
			message.append("==================================\n");
			std::cout << message << std::endl;
			FileSystem::write_all_text(path, message);
		}
	}
	void log(string message, LogLevel level)
	{
		if(m_running.load())
		{
			if(level < verbosity) return;
			std::string log_message = "";
			timer.now();
			log_message.append(timer.get_log_string());
			log_message.append(log_levels[level]);
			log_message.append(message);
			log_message.append("\n");
			log_queue->push(log_message);
		}
	}
};


#endif