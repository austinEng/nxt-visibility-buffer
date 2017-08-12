
#pragma once

#include <mutex>

template <typename Object>
class LockedObject {
    public:
        LockedObject() { }
        LockedObject(const Object &other) : object(other.Clone()) { }
        LockedObject& operator=(const Object& other) { object = other.Clone(); return *this; }
        LockedObject(Object&& other) : object(std::move(other)) { }
        LockedObject& operator=(Object&& other) { object = std::move(other); return *this; }

        LockedObject(const LockedObject& other) : object(other.object.Clone()) { }
        LockedObject& operator=(const LockedObject& other) { object = other.object.Clone(); return *this; }
        LockedObject(LockedObject&& other) : object(std::move(other.object)) { }
        LockedObject& operator=(LockedObject&& other) { object = std::move(other.object); return *this; }

        Object& Get() {
            return object;
        }

        Object& Lock() {
            mutex.lock();
            return object;
        }

        void Unlock() {
            mutex.unlock();
        }

    private:
        std::mutex mutex;
        Object object;
};

#define LOCK_AND_RELEASE(object, body) [&](){ auto& o = object.Lock().body; object.Unlock(); return std::move(o); }()
#define LOCK_AND_RELEASE_VOID(object, body) [&](){ object.Lock().body; object.Unlock(); }()
