#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include <string>
#include <memory>
#include <random>
#include <chrono>
#include <vector>
#include <deque>
#include <unordered_map>
#include <algorithm>

using namespace std;

#define sptr shared_ptr
#define wptr weak_ptr
#define COMPONENT_ID size_t

class Component : public enable_shared_from_this<Component>
{
protected:
    COMPONENT_ID id;
    uint64_t depth;
    wptr<Component> parent;
    string name;
    string type;
private:
    vector<COMPONENT_ID> order;
    unordered_map<COMPONENT_ID, sptr<Component>> child_map;
    unordered_map<COMPONENT_ID, wptr<Component>> weak_map;
    unordered_map<string, COMPONENT_ID> name_map;
public:
    Component();
    ~Component();
public:
    template<typename T> wptr<T> add_component();
    template<typename T> wptr<T> add_component(string _name, bool unique_name = true);
    template<typename T> void link_component(wptr<T>& _target_module);
    // Find Component with ID (recursive = »óÀ§ ¸ðµç °´Ã¼ Å½»ö(´ÜÀÏ°´Ã¼ Å½»ö, Tree ¾Æ´Ô)
    template<typename T> wptr<T> find_component(COMPONENT_ID _id, bool recursive_search = false);
    // Find Component with Component Name (recursive = »óÀ§ ¸ðµç °´Ã¼ Å½»ö(´ÜÀÏ°´Ã¼ Å½»ö, Tree ¾Æ´Ô)
    template<typename T> wptr<T> find_component(string _name, bool recursive_search = false);
    // Find Components with Component Type(Ordered)
    template<typename T> deque<wptr<T>> find_components_with_type(string _type);
    template<typename T> wptr<T> get_parent();
public:
    void delete_component(COMPONENT_ID _id);
    void delete_component(string _name);
    uint64_t get_component_depth();
    COMPONENT_ID get_component_id();
    string get_component_name();
    void set_component_name(string _name);
    string get_component_type();
    void set_component_type(string _type);
};
#include "Component.hpp"
#endif