package com.a303.notificationms.domain.memberToken.repository;

import com.a303.notificationms.domain.memberToken.MemberToken;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface MemberTokenRepository extends MongoRepository<MemberToken, String>, MemberTokenCustomRepository {
    MemberToken findMemberTokenById(String id);

}
