# Self-Localization

---

## Objective ![progress 70%](https://progress-bar.dev/70/?scale=100&title=)

Achieve a system that can <mark>localize itself using a vision sensor (monocular camera) extracted (multiple-dissimilar) features/objects</mark> in addition to motion sensor (IMU) in indoor and/or outdoor environment.

---

## Scope

Make use of a vision sensor in cooperation of a motion sensor to sustain vehicle localization

---

## Submodule(s)

> ### [ALIKED](https://github.com/Shiaoming/ALIKED)
>>
>> #### Installation On Windows
>> 0- You can skip the following (1-4) steps if you know git patches, you can apply the patch `git_patch_ALIKED.diff` into `ALIKED` submodule after cloning it
>>
>> 1- Browse to `ALIKED\custom_ops` folder location (ex: `cd ALIKED\custom_ops`)
>> 
>> 2- Modify `__init__.py` replace extension from `.so` to `.pyd`
>> 
>> ```Python
>> ...
>>     for f in file_path.parent.glob('get_patches*.pyd'): # <<< Here replace `.so` to `.pyd`
>> ...
>> ```
>> 
>> 3- Modify `get_patches.cpp` uncomment `PYBIND11_MODULE(get_patches, m)` block
>> 
>> ```C++
>> PYBIND11_MODULE(get_patches, m)
>> {
>>  m.doc() = "Get patches for a CxHxW map of Nx2 locations.";
>> 
>>  m.def("get_patches_forward", &get_patches_forward, "get_patches forward");
>>  m.def("get_patches_backward", &get_patches_backward, "get_patches backward");
>> }
>> ```
>> 
>> 4- Modify `get_patches.cpp` replace `long` with `long long`
>> 
>> ```C++
>> ...
>>     // get patches
>>     torch::Tensor patches = torch::zeros({N, C, kernel_size, kernel_size}, map.options());
>>     auto a_points = points.accessor<long long, 2>();    // Nx2 // <<< Here
>>     auto a_map_pad = map_pad.accessor<float, 3>(); // Cx(H+2*radius)x(W+2*radius)
>> ...
>>     torch::Tensor d_map_pad = torch::zeros({C, H + int(2 * radius), W + int(2 * radius)}, d_patches.options());
>> 
>>     auto a_points = points.accessor<long long, 2>();        // Nx2 // <<< Here
>>     auto a_d_map_pad = d_map_pad.accessor<float, 3>(); // Cx(H+2*radius)x(W+2*radius)
>> ...
>> ```
>> 
>> 5- Install `wheel` (ex: `python -m pip install wheel`)
>> 
>> 6- Install `custom_ops` (ex: `python -m pip install .`)
>> 
>> 7- If everything went fine, then you're done installing ALIKED pre-requisites
>> 

>> ##### Known Issue(s) FIX

>>> - torch `Tuple` compilation error

>>>> Modify `torch\include\pybind11\cast.h` class `tuple_caster` copy class definition to be after the `type_caster` for pair and tuple

>>>> ```C++
>>>> // Base implementation for std::tuple and std::pair
>>>> template <template<typename...> class Tuple, typename... Ts> class tuple_caster; // <<< keep prototype here
>>>> template <typename T1, typename T2> class type_caster<std::pair<T1, T2>>
>>>>     : public tuple_caster<std::pair, T1, T2> {};
>>>> template <typename... Ts> class type_caster<std::tuple<Ts...>>
>>>>     : public tuple_caster<std::tuple, Ts...> {};
>>>> template <template<typename...> class Tuple, typename... Ts> class tuple_caster // <<< move definition here
>>>> ```

>>> - Compiled `custom_ops*.pyd` libary is not seen for installation

>>>> Copy `custom_ops*.pyd` into same folder that has `__init__.py`


> ### [silk](https://github.com/facebookresearch/silk)
>>
>> #### Installation On Windows
>> 0- You can skip the following (...) steps if you know git patches, you can apply the patch `git_patch_silk.diff` into `silk` submodule after cloning it
>>

---

## Reference(s)

- [REF](https://)

## Resource(s)

- [](https://progress-bar.dev/100/?scale=100&title=progress)
- [](https://github.com/fredericojordan/progress-bar)
- [](https://www.markdownguide.org/basic-syntax/)
- [](https://daringfireball.net/projects/markdown/syntax)
