#ifndef __MEMORY_OBJECT_POOL_HPP__
#define __MEMORY_OBJECT_POOL_HPP__

#include <mutex>
#include <list>
#include "../Library/onetbb_win/include/tbb/concurrent_queue.h"

using namespace tbb::detail::d2;

template<typename T>
class MemoryObjectPool
{
private:
    std::mutex mtx;
    T* origin_pool;
    int pool_size;
    concurrent_queue<T*>* pool;
public:
    MemoryObjectPool()
    {
        pool = new concurrent_queue<T*>();
        origin_pool = nullptr;
        pool_size = 0;
    }
    ~MemoryObjectPool()
    {
        delete pool;
    }
    void initialize(int init_size = 32)
    {
        std::unique_lock<std::mutex> locker(mtx);
        this->pool_size = init_size;
        origin_pool = new T[this->pool_size];
        for (int i = 0; i < this->pool_size; i++)
        {
            T* obj = &origin_pool[i];
            pool->push(obj);
        }
    }
    void release()
    {
        std::unique_lock<std::mutex> locker(mtx);
        pool->clear();
        delete[] origin_pool;
        this->pool_size = 0;
    }
    void debug_print()
    {
        printf("object_pool : %d\n", pool_size);
    }
public:
    T* spawn()
    {
        while (true)
        {
            if (pool->empty() == true) break;
            T* obj;
            if (pool->try_pop(obj))
            {
                return obj;
            }
        }
        return nullptr;
    }
    void despawn(T* obj)
    {
        pool->push(obj);
    }
};

#endif