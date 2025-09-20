"""
    RosenblattTransformation <: AbstractTransformation

Type used to perform Rosenblatt Transformation.

$(TYPEDFIELDS)
"""
mutable struct RosenblattTransformation <: AbstractIsoprobabilisticTransformation

end
Base.broadcastable(x::RosenblattTransformation) = Ref(x)