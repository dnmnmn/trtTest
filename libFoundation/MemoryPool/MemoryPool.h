#ifndef __MEMORY_POOL_H__
#define __MEMORY_POOL_H__

#include <vector>
#include <unordered_map>
#include <mutex>
#include <queue>

using namespace std;

template<typename T>
class MemoryPool
{
public:
    class MemoryCell
    {
        friend class MemoryPool;
    private:
        T* internal_data;
        unsigned int internal_len;
        unsigned int current_len;
    public:
        unsigned int len()
        {
            return current_len;
        }
        T* data()
        {
            return internal_data;
        }
    };
private:
    std::mutex mtx;
    std::unordered_map<unsigned int, queue<MemoryCell*>> origin_pool;
    std::unordered_map<unsigned int, queue<MemoryCell*>> pool;
public:
    void initialize()
    {
        //std::unique_lock<std::mutex> locker(mtx);

    }
    void release()
    {
        std::unique_lock<std::mutex> locker(mtx);
        pool.clear();
        auto itr = origin_pool.begin();
        while (itr != origin_pool.end())
        {
            while (true)
            {
                delete[] itr->second.front()->internal_data;
                itr->second.pop();
                if (itr->second.size() <= 0) break;
            }
            itr++;
        }
        origin_pool.clear();
    }
public:
    MemoryCell* spawn(unsigned int length)
    {
        std::unique_lock<std::mutex> locker(mtx);
        unsigned int target_len = 1;
        while (true)
        {
            target_len *= 2;
            if (target_len >= length) break;
        }
        auto pool_itr = pool.find(target_len);
        if (pool_itr == pool.end())
        {
            // queue 추가
            queue<MemoryCell*> new_list;
            pool.insert(pair<unsigned int, queue<MemoryCell*>>(target_len, new_list));
            pool_itr = pool.find(target_len);
        }
        if (pool_itr->second.size() <= 0)
        {
            // memory cell 추가
            MemoryCell* cell = new MemoryCell();
            cell->internal_len = target_len;
            cell->internal_data = new T[target_len];
            cell->current_len = length;
            auto itr = origin_pool.find(target_len);
            if (itr == origin_pool.end())
            {
                queue<MemoryCell*> new_list;
                origin_pool.insert(pair<unsigned int, queue<MemoryCell*>>(target_len, new_list));
                itr = origin_pool.find(target_len);
            }
            itr->second.push(cell);
            pool_itr->second.push(cell);

            /*printf("===============================\n");
            auto tmp = origin_pool.begin();
            while (tmp != origin_pool.end())
            {
                printf("pool %d key / %d objs\n", tmp->first, tmp->second.size());
                tmp++;
            }*/
        }
        MemoryCell* obj = pool_itr->second.front();
        pool_itr->second.pop();
        obj->current_len = length;

        //printf("pool : %d\n", origin_pool.size());
        return obj;
    }
    void despawn(MemoryCell* addr)
    {
        std::unique_lock<std::mutex> locker(mtx);
        auto itr = pool.find(addr->internal_len);
        if (itr != pool.end())
        {
            itr->second.push(addr);
        }
    }
};

#endif