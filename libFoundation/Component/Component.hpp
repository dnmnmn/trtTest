template<typename T>
wptr<T> Component::add_component()
{
    return add_component<T>("noname", false);
}

template<typename T>
wptr<T> Component::add_component(string _name, bool unique_name)
{
    sptr<T> comp = make_shared<T>();
    wptr<T> weak = comp;
    COMPONENT_ID target_id = comp->get_component_id();
    string new_name = _name;
    if (unique_name == true)
    {
        int cnt = 0;
        while (true)
        {
            bool flag = false;
            auto itr = name_map.find(new_name);
            if (itr != name_map.end())
            {
                flag = true;
            }
            if (flag == true)
            {
                new_name = _name;
                new_name.append("_");
                new_name.append(to_string(cnt));
                cnt++;
            }
            else break;
        }
    }
    comp->name = new_name;
    comp->parent = shared_from_this();
    //find depth
    depth = 1;
    wptr<Component> depth_target = comp->parent.lock();
    while (true)
    {
        if (depth_target.lock() == nullptr) break;
        depth_target = depth_target.lock()->parent.lock();
        depth++;
    }
    order.push_back(comp->get_component_id());
    name_map.insert(pair<string, COMPONENT_ID>(comp->name, target_id));
    child_map.insert(pair<COMPONENT_ID, sptr<Component>>(target_id, std::move(comp)));
    auto f = child_map.find(target_id)->second;
    weak = static_pointer_cast<T>(f);
    weak_map.insert(pair<COMPONENT_ID, wptr<Component>>(target_id, weak));
    return weak;
}

template<typename T>
void Component::link_component(wptr<T>& _target_module)
{
    COMPONENT_ID target_id = _target_module.lock()->get_component_id();
    order.push_back(target_id);
    // weak map ¿¬°á
    weak_map.insert(pair<COMPONENT_ID, wptr<Component>>(target_id, _target_module));
    name_map.insert(pair<string, COMPONENT_ID>(_target_module.lock()->get_name(), target_id));
}

template<typename T>
wptr<T> Component::find_component(COMPONENT_ID _id, bool recursive_search)
{
    if (recursive_search == false)
    {
        auto itr = weak_map.find(_id);
        if (itr != weak_map.end()) return static_pointer_cast<T>(itr->second.lock());
    }
    else
    {
        sptr<Component> search_target = shared_from_this();
        while (true)
        {
            auto itr = search_target->weak_map.find(_id);
            if (itr != search_target->weak_map.end())
            {
                return static_pointer_cast<T>(itr->second.lock());
            }
            else
            {
                search_target = search_target->get_parent<Component>().lock();
                if (!search_target) break;
            }
        }
    }
    return wptr<T>();
}

template<typename T>
wptr<T> Component::find_component(string _name, bool recursive_search)
{
    if (recursive_search == false)
    {
        auto itr = weak_map.begin();
        while (itr != weak_map.end())
        {
            if (itr->second.lock()->name == _name)
            {
                return static_pointer_cast<T>(itr->second.lock());
            }
            itr++;
        }
    }
    else
    {
        sptr<Component> search_target = shared_from_this();
        auto itr = search_target->weak_map.begin();
        while (true)
        {
            if (search_target->name == _name)
            {
                return static_pointer_cast<T>(search_target);
            }
            itr = search_target->weak_map.begin();
            while (itr != search_target->weak_map.end())
            {
                if (itr->second.lock()->name == _name)
                {
                    return static_pointer_cast<T>(itr->second.lock());
                }
                itr++;
            }
            search_target = search_target->get_parent<Component>().lock();
            if (!search_target) break;
        }
    }
    return wptr<T>();
}

template<typename T>
deque<wptr<T>> Component::find_components_with_type(string _type)
{
    deque<wptr<T>> vec;
    auto itr = order.begin();
    while (itr != order.end())
    {
        auto weak = weak_map.find((*itr));
        if (weak->second.lock()->type.find(_type) != string::npos)
        {
            vec.push_back(static_pointer_cast<T>(weak->second.lock()));
        }
        else
        {
            auto vec_child = weak->second.lock()->find_components_with_type<T>(_type);
            for (int i = 0; i < vec_child.size(); i++) vec.push_back(vec_child[i]);
        }
        itr++;
    }
    return vec;
}

template<typename T>
wptr<T> Component::get_parent()
{
    return dynamic_pointer_cast<T>(parent.lock());
}