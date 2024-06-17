#include "Component.h"

// 생성자
Component::Component()
{
    id = COMPONENT_ID(this);
    set_component_name("basic_component");
    set_component_type("basic_component");
    parent.reset();
}

// 소멸자
Component::~Component()
{
    auto weak = weak_map.begin();
    while (weak != weak_map.end())
    {
        auto child = child_map.find(weak->first);
        if (child != child_map.end())
        {
            child->second.reset();
            child->second = nullptr;
            child = child_map.erase(child);
        }
        weak->second.reset();
        weak = weak_map.erase(weak);
    }
    parent.reset();

    child_map.clear();
    weak_map.clear();
    name_map.clear();
}

void Component::delete_component(COMPONENT_ID _id)
{
    auto weak = weak_map.find(_id);
    if (weak != weak_map.end())
    {
        // name map에서 제거
        auto np = name_map.find(weak->second.lock()->get_component_name());
        if (np != name_map.end())
        {
            np = name_map.erase(np);
        }
        // Erase
        auto order_itr = order.begin();
        while (order_itr != order.end())
        {
            if ((*order_itr) == weak->first)
            {
                order.erase(order_itr);
                break;
            }
            order_itr++;
        }
        order.shrink_to_fit();
        weak->second.reset();
        weak = weak_map.erase(weak);
        // Child map 에서 원본 제거
        auto comp = child_map.find(_id);
        if (comp != child_map.end())
        {
            comp->second.reset();
            comp->second = nullptr;
            comp = child_map.erase(comp);
        }
    }
}

void Component::delete_component(string _name)
{
    auto weak = weak_map.begin();
    while (weak != weak_map.end())
    {
        auto sptr_comp = weak->second.lock();
        if (sptr_comp->name == _name)
        {
            // name map에서 제거
            auto np = name_map.find(sptr_comp->get_component_name());
            if (np != name_map.end())
            {
                np = name_map.erase(np);
            }
            //Erase
            auto order_itr = order.begin();
            while (order_itr != order.end())
            {
                if ((*order_itr) == weak->first)
                {
                    order.erase(order_itr);
                    break;
                }
                order_itr++;
            }
            order.shrink_to_fit();
            // Child map 에서 원본 제거
            auto comp = child_map.find(weak->first);
            if (comp != child_map.end())
            {
                comp->second.reset();
                comp->second = nullptr;
                comp = child_map.erase(comp);
            }
            weak->second.reset();
            weak = weak_map.erase(weak);
            continue;
        }
        weak++;
    }
}

uint64_t Component::get_component_depth()
{
    return depth;
}

COMPONENT_ID Component::get_component_id()
{
    return id;
}

string Component::get_component_name()
{
    return name;
}

void Component::set_component_name(string _name)
{
    this->name = _name;
}

string Component::get_component_type()
{
    return type;
}

void Component::set_component_type(string _type)
{
    this->type = _type;
}